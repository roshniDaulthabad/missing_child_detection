import cv2
import numpy as np
import time
from app.config import Config

class FaceEnhancer:
    """Modular Super-Resolution & Face Enhancer optimized for CPU.
    Selectively enhances marginal or low-resolution face crops without choking CPU inference.
    """

    def __init__(self, enabled: bool = None):
        active_settings = Config.get_active_mode_settings()
        self.enabled = enabled if enabled is not None else active_settings.get("enable_enhancement", True)
        self.model_name = "CPU-Bicubic-Unsharp-EDSR-v1"

    def should_enhance(self, face_bgr: np.ndarray, quality_score: float) -> bool:
        """Determines if the face crop warrants super-resolution enhancement."""
        if not self.enabled or face_bgr is None or face_bgr.size == 0:
            return False
        h, w = face_bgr.shape[:2]
        # Only invoke if face resolution is marginal (< 80px) and quality is between 0.35 and 0.75
        if (w < 80 or h < 80) and 0.35 <= quality_score <= 0.75:
            return True
        return False

    def enhance_face(self, face_bgr: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        """Enhances face crop using high-order Lanczos/cubic resampling followed by
        adaptive unsharp masking to recover facial edge structure on CPU.
        """
        if face_bgr is None or face_bgr.size == 0:
            return face_bgr

        start = time.time()
        h, w = face_bgr.shape[:2]
        target_w = w * scale_factor
        target_h = h * scale_factor

        # 1. High-order interpolation
        upscaled = cv2.resize(face_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # 2. Adaptive unsharp mask to restore edge definition
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        sharpened = cv2.addWeighted(upscaled, 1.4, gaussian, -0.4, 0)

        # 3. Subtle contrast-limited adaptive histogram equalization (CLAHE) on luminance
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        duration_ms = (time.time() - start) * 1000.0
        return enhanced

face_enhancer_service = FaceEnhancer()
