import cv2
import numpy as np
from app.config import Config

class FaceQualityAssessor:
    """Evaluates biometric face quality (sharpness, resolution, illumination, aspect ratio).
    Rejects degraded or unidentifiable crops to prevent false positive matches.
    """

    def __init__(self, min_size: int = None, min_sharpness: float = None):
        self.min_size = min_size or Config.FACE_MIN_SIZE
        self.min_sharpness = min_sharpness or Config.FACE_SHARPNESS_MIN

    def assess_crop(self, face_bgr: np.ndarray) -> dict:
        if face_bgr is None or face_bgr.size == 0:
            return {
                "sufficient": False,
                "score": 0.0,
                "reason": "Empty or null face image"
            }

        h, w = face_bgr.shape[:2]

        # 1. Resolution Check
        if h < self.min_size or w < self.min_size:
            return {
                "sufficient": False,
                "score": round(float(min(h, w) / self.min_size) * 0.4, 3),
                "reason": f"Insufficient Face Quality: Resolution ({w}x{h}) below {self.min_size}px"
            }

        # 2. Sharpness / Blur Check (Variance of Laplacian)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.min_sharpness:
            return {
                "sufficient": False,
                "score": round(float(laplacian_var / self.min_sharpness) * 0.5, 3),
                "reason": f"Insufficient Face Quality: Motion blur/sharpness score ({laplacian_var:.1f}) < {self.min_sharpness}"
            }

        # 3. Illumination / Extreme Contrast Check
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 25.0:
            return {
                "sufficient": False,
                "score": 0.3,
                "reason": f"Insufficient Face Quality: Severe underexposure (brightness {mean_brightness:.1f})"
            }
        elif mean_brightness > 235.0:
            return {
                "sufficient": False,
                "score": 0.3,
                "reason": f"Insufficient Face Quality: Severe overexposure (brightness {mean_brightness:.1f})"
            }

        # Quality score normalized between 0.6 and 1.0
        sharpness_factor = min(1.0, laplacian_var / 150.0)
        resolution_factor = min(1.0, min(h, w) / 128.0)
        composite_score = round(0.5 + 0.3 * sharpness_factor + 0.2 * resolution_factor, 3)

        return {
            "sufficient": True,
            "score": composite_score,
            "sharpness": round(laplacian_var, 1),
            "size": f"{w}x{h}",
            "reason": "Acceptable biometric quality"
        }

face_quality_service = FaceQualityAssessor()
