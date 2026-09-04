import os
import cv2
import numpy as np
from pathlib import Path
from app.config import Config

class FaceDetector:
    """High-speed CPU face detector using YuNet ONNX with heuristic CCTV head-region fallback."""

    def __init__(self, model_path: str = None, score_thresh: float = 0.5):
        default_model = Path(Config.MODELS_DIR) / "face" / "face_detection_yunet_2023mar.onnx"
        self.model_path = model_path or str(default_model)
        self.score_thresh = score_thresh
        self.detector = None

        if os.path.exists(self.model_path):
            try:
                self.detector = cv2.FaceDetectorYN_create(
                    model = self.model_path,
                    config = "",
                    input_size = (320, 320),
                    score_threshold = self.score_thresh,
                    nms_threshold = 0.3,
                    top_k = 10
                )
            except Exception as e:
                print(f"[WARNING] Failed to initialize YuNet detector: {e}. Using head-crop fallback.")
                self.detector = None

    def detect_faces(self, image_bgr: np.ndarray) -> list:
        """Detects faces in an image (or person crop).
        Returns list of dicts: [{'bbox': [x1, y1, x2, y2], 'conf': float, 'crop': np.ndarray, 'landmarks': [...]}]
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        results = []

        if self.detector is not None and w >= 32 and h >= 32:
            try:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(image_bgr)
                if faces is not None:
                    for f in faces:
                        # YuNet format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
                        fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        conf = float(f[14])
                        x1 = max(0, fx)
                        y1 = max(0, fy)
                        x2 = min(w, fx + fw)
                        y2 = min(h, fy + fh)
                        if x2 > x1 and y2 > y1:
                            crop = image_bgr[y1:y2, x1:x2].copy()
                            landmarks = f[4:14].reshape(5, 2).tolist()
                            results.append({
                                "bbox": [x1, y1, x2, y2],
                                "conf": round(conf, 3),
                                "crop": crop,
                                "landmarks": landmarks,
                                "source": "yunet"
                            })
            except Exception as e:
                pass

        # If no face found by YuNet (e.g. side profile, looking down, back-turned),
        # extract the anatomical head region from person crop (top 25% of height)
        if not results and h > 80 and w > 40:
            head_h = int(h * 0.28)
            head_margin_x = int(w * 0.15)
            hx1 = head_margin_x
            hy1 = 0
            hx2 = w - head_margin_x
            hy2 = head_h
            if hx2 > hx1 and hy2 > hy1:
                crop = image_bgr[hy1:hy2, hx1:hx2].copy()
                results.append({
                    "bbox": [hx1, hy1, hx2, hy2],
                    "conf": 0.45,
                    "crop": crop,
                    "landmarks": None,
                    "source": "anatomical_cctv_head"
                })

        return results

face_detector_service = FaceDetector()
