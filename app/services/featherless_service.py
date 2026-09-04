import os
import cv2
import json
import re
import base64
import requests
from typing import Dict, Any, Optional
from app.config import Config

class FeatherlessService:
    """Integrates Featherless.ai multimodal vision LLM inference (Qwen/Qwen3.8-Flash-Next)
    to automatically verify and assess potential missing child biometric alerts.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else Config.FEATHERLESS_API_KEY
        self.base_url = (base_url or Config.FEATHERLESS_BASE_URL).rstrip('/')
        self.model = model or Config.FEATHERLESS_MODEL
        self.timeout = 25  # Timeout in seconds for multimodal vision inference

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def _encode_image_b64(self, path: Optional[str], max_dim: int = 256) -> Optional[str]:
        """Encodes an image file to base64 JPEG format, resizing to max_dim for fast transfer."""
        if not path:
            return None
        candidate_paths = [path, os.path.join(Config.BASE_DIR, path)]
        valid_path = None
        for p in candidate_paths:
            if os.path.exists(p) and os.path.isfile(p):
                valid_path = p
                break
        if not valid_path:
            return None

        try:
            img = cv2.imread(valid_path)
            if img is None:
                return None
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception as e:
            print(f"[FEATHERLESS] Error encoding image {path}: {e}")
            return None

    def assess_potential_match(self, child_data: Dict[str, Any], alert_data: Dict[str, Any]) -> Dict[str, str]:
        """Sends biometric detection, visual images, and missing child case metadata to Featherless.ai
        using the multimodal vision model Qwen/Qwen3.8-Flash-Next.
        Guarantees fallback without raising exceptions if offline or failing.
        """
        score_val = alert_data.get('similarity_score', 0.0)
        try:
            score_pct = float(score_val)
            if score_pct <= 1.0:
                score_pct = score_pct * 100.0
        except (ValueError, TypeError):
            score_pct = 0.0

        if not self.is_configured():
            return {
                "verdict": "Under Review",
                "status": "Under Review (Offline Fallback)",
                "confidence": "Pending Officer Review",
                "reasoning": f"Featherless API key not configured. Relying on primary AdaFace biometric match ({score_pct:.1f}%).",
                "recommendation": "Perform manual side-by-side visual comparison in the verification portal."
            }

        # Check for visual images (registered reference photo and CCTV face crop)
        ref_path = alert_data.get("reference_photo_path") or child_data.get("photo_path")
        face_path = alert_data.get("face_crop_path")

        ref_b64 = self._encode_image_b64(ref_path)
        face_b64 = self._encode_image_b64(face_path)
        has_images = bool(ref_b64 and face_b64)

        prompt = self._build_investigative_prompt(child_data, alert_data, score_pct, has_images=has_images)

        if has_images:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{face_b64}"}}
            ]
        else:
            user_content = prompt

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "EasyFind-App/1.0"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "temperature": 0.2,
            "max_tokens": 512
        }

        try:
            endpoint = f"{self.base_url}/chat/completions"
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
            
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return self._parse_llm_json(content, score_pct)
            else:
                err_snippet = resp.text[:140]
                print(f"[FEATHERLESS] API returned HTTP {resp.status_code}: {err_snippet}")
                return self._build_fallback(f"HTTP {resp.status_code}", score_pct)

        except requests.exceptions.Timeout:
            print(f"[FEATHERLESS] API request timed out (exceeded {self.timeout}s)")
            return self._build_fallback("Request timed out", score_pct)
        except requests.exceptions.RequestException as e:
            print(f"[FEATHERLESS] Network connection error: {e}")
            return self._build_fallback("Network error", score_pct)
        except Exception as e:
            print(f"[FEATHERLESS] Unexpected error during verification: {e}")
            return self._build_fallback(str(e), score_pct)

    def _build_investigative_prompt(self, child: Dict[str, Any], alert: Dict[str, Any], score_pct: float, has_images: bool = False) -> str:
        case_id = child.get("case_id", "Unknown")
        name = child.get("child_name", "Unknown")
        age = child.get("age", "Unknown")
        gender = child.get("gender", "Unknown")
        date_missing = child.get("date_missing", "Unknown")
        last_loc = child.get("last_known_location", "Unknown")
        desc = child.get("physical_description", "N/A")
        ident = child.get("identifying_characteristics", "N/A")
        priority = child.get("priority", "High")

        cam_id = alert.get("camera_id", "CAM-01")
        loc = alert.get("camera_location", "Surveillance Stream")
        track_id = alert.get("track_id", 0)
        time_str = alert.get("timestamp", "Recent")

        image_intro = ""
        if has_images:
            image_intro = (
                "VISUAL OBSERVATIONS ATTACHED:\n"
                "- Image 1: Registered primary reference photo of the missing child.\n"
                "- Image 2: CCTV biometric face crop detected on camera.\n\n"
            )

        return (
            f"You are evaluating a missing child alert generated by an automated CCTV biometric system.\n\n"
            f"{image_intro}"
            f"MISSING CHILD RECORD:\n"
            f"- Case ID: {case_id}\n"
            f"- Name: {name}, Age: {age}, Gender: {gender}\n"
            f"- Missing Date: {date_missing}\n"
            f"- Last Known Location: {last_loc}\n"
            f"- Physical Description: {desc}\n"
            f"- Identifying Features: {ident}\n"
            f"- Priority: {priority}\n\n"
            f"CCTV SIGHTING:\n"
            f"- Camera: {cam_id} ({loc})\n"
            f"- Time of sighting: {time_str}\n"
            f"- AdaFace Biometric Face Similarity: {score_pct:.1f}%\n"
            f"- CCTV Person Track ID: #{track_id}\n\n"
            f"TASK:\n"
            f"Assess if this sighting matches the missing child based on visual facial comparison (if images attached), biometric similarity score, location proximity, and child profile.\n"
            f"Provide an automated verification decision (verdict):\n"
            f"- If biometric similarity is high (>=70%) and facial/case details align, verdict is 'Confirm'.\n"
            f"- If biometric similarity is low or facial features clearly conflict, verdict is 'Reject'.\n"
            f"- If inconclusive or requires immediate manual field inspection, verdict is 'Under Review'.\n\n"
            f"Return ONLY a JSON object with exactly these keys:\n"
            f'- "verdict": choose one ("Confirm", "Reject", "Under Review")\n'
            f'- "status": choose one ("Confirmed by Featherless AI", "Rejected by Featherless AI", "Under Review (Featherless AI)")\n'
            f'- "confidence": choose one ("High", "Medium", "Low")\n'
            f'- "reasoning": 2-3 concise sentences analyzing visual facial alignment, biometric match, and case details.\n'
            f'- "recommendation": 1-2 actionable operational steps for investigating officers.'
        )

    def _parse_llm_json(self, raw_text: str, score_pct: float) -> Dict[str, str]:
        clean_text = raw_text.strip()
        # Regex search for JSON block
        json_match = re.search(r"\{[\s\S]*\}", clean_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                verdict = parsed.get("verdict", "")
                status = parsed.get("status", "")
                confidence = parsed.get("confidence", "Medium")
                reasoning = parsed.get("reasoning", "AI biometric evaluation completed.")
                recommendation = parsed.get("recommendation", "Officer visual review advised.")

                v_lower = str(verdict).lower()
                s_lower = str(status).lower()
                if "confirm" in v_lower or "confirm" in s_lower or "likely" in s_lower:
                    normalized_status = "Confirmed by Featherless AI"
                    norm_verdict = "Confirm"
                elif "reject" in v_lower or "reject" in s_lower or "low" in s_lower:
                    normalized_status = "Rejected by Featherless AI"
                    norm_verdict = "Reject"
                else:
                    normalized_status = "Under Review (Featherless AI)"
                    norm_verdict = "Under Review"

                return {
                    "verdict": norm_verdict,
                    "status": normalized_status,
                    "confidence": str(confidence),
                    "reasoning": str(reasoning),
                    "recommendation": str(recommendation)
                }
            except Exception:
                pass

        # Fallback if no valid JSON object
        if score_pct >= 70.0:
            norm_status = "Confirmed by Featherless AI"
            norm_verdict = "Confirm"
            conf = "High" if score_pct >= 85.0 else "Medium"
        elif score_pct < 50.0:
            norm_status = "Rejected by Featherless AI"
            norm_verdict = "Reject"
            conf = "Low"
        else:
            norm_status = "Under Review (Featherless AI)"
            norm_verdict = "Under Review"
            conf = "Medium"

        return {
            "verdict": norm_verdict,
            "status": norm_status,
            "confidence": conf,
            "reasoning": clean_text[:280] if len(clean_text) > 20 else f"Biometric face similarity calculated at {score_pct:.1f}%.",
            "recommendation": "Verify CCTV face crop against reference photo before dispatching officers."
        }

    def _build_fallback(self, reason: str, score_pct: float) -> Dict[str, str]:
        conf = "High" if score_pct >= 85.0 else ("Medium" if score_pct >= 70.0 else "Low")
        status = "Under Review (Offline Fallback)"
        return {
            "verdict": "Under Review",
            "status": status,
            "confidence": conf,
            "reasoning": f"Featherless AI service unavailable ({reason}). Primary AdaFace biometric score: {score_pct:.1f}%.",
            "recommendation": "Perform manual visual verification using the side-by-side review tool."
        }
featherless_service = FeatherlessService()