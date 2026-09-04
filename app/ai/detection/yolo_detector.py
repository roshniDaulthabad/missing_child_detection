import time
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from app.config import Config
from app.ai.hardware import RealTimeProfiler

class YOLOPersonDetector:
    """YOLO11n Person Detector operating strictly on CPU.
    Detects humans (COCO class 0: person) with configurable confidence and image resolution.
    """

    def __init__(self, model_path: str = None, conf_threshold: float = None, imgsz: int = None):
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        if not os.path.exists(self.model_path):
            # Fallback to local yolo11n.pt if exists
            local_yolo = Path("models/yolo/yolo11n.pt")
            if local_yolo.exists():
                self.model_path = str(local_yolo)
            else:
                self.model_path = "yolo11n.pt"

        # Load YOLO model strictly on CPU
        self.model = YOLO(self.model_path)
        self.conf_threshold = conf_threshold or Config.get_active_mode_settings()["conf_threshold"]
        self.imgsz = imgsz or Config.get_active_mode_settings()["imgsz"]
        self.device = "cpu"
        self.profiler = RealTimeProfiler()

    def detect_persons(self, frame: np.ndarray):
        """Runs person detection on a single frame (BGR numpy array).
        Returns a list of dicts: [{'bbox': [x1, y1, x2, y2], 'conf': float, 'crop': np.ndarray}]
        """
        start = time.time()
        # Run inference strictly on CPU, filtering class 0 (person)
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device="cpu",
            verbose=False
        )
        duration = time.time() - start
        self.profiler.tick(duration)

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            h, w = frame.shape[:2]
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
                crop = frame[y1:y2, x1:x2].copy() if (y2 > y1 and x2 > x1) else None
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": round(conf, 3),
                    "crop": crop
                })

        return detections

    def get_metrics(self):
        return self.profiler.get_summary()
