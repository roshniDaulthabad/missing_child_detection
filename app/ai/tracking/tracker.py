import time
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from app.config import Config
from app.ai.hardware import RealTimeProfiler

class TrackHistory:
    """Maintains trajectory and state for a single persistent track."""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.trajectory = []  # list of (cx, cy) centroids
        self.last_bbox = None
        self.frames_since_face_check = 999  # Force check on first appearance
        self.last_face_match = None
        self.appearance_crops = []

    def update(self, bbox: list, crop: np.ndarray, interval: int) -> bool:
        self.last_seen = time.time()
        self.last_bbox = bbox
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > 50:
            self.trajectory.pop(0)

        if crop is not None and len(self.appearance_crops) < 5:
            self.appearance_crops.append(crop)

        self.frames_since_face_check += 1
        # Trigger face analysis only every `interval` frames
        if self.frames_since_face_check >= interval:
            self.frames_since_face_check = 0
            return True
        return False


class PersonTracker:
    """Multi-object person tracker using BoT-SORT / ByteTrack strictly on CPU."""

    def __init__(self, model_path: str = None, tracker_type: str = None):
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        if not os.path.exists(self.model_path):
            local_yolo = Path("models/yolo/yolo11n.pt")
            self.model_path = str(local_yolo) if local_yolo.exists() else "yolo11n.pt"

        self.model = YOLO(self.model_path)
        active_settings = Config.get_active_mode_settings()
        self.tracker_cfg = tracker_type or active_settings.get("tracker", "botsort.yaml")
        self.imgsz = active_settings.get("imgsz", 512)
        self.conf = active_settings.get("conf_threshold", 0.45)
        self.interval = active_settings.get("face_recognition_interval", 10)
        self.tracks = defaultdict(lambda: None)
        self.profiler = RealTimeProfiler()

    def update(self, frame: np.ndarray):
        """Processes a frame, tracks all persons, and updates track trajectories.
        Returns: list of tracked objects dicts.
        """
        start = time.time()
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[0],
            tracker=self.tracker_cfg,
            conf=self.conf,
            imgsz=self.imgsz,
            device="cpu",
            verbose=False
        )
        duration = time.time() - start
        self.profiler.tick(duration)

        tracked_objects = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            h, w = frame.shape[:2]

            for box in boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
                crop = frame[y1:y2, x1:x2].copy() if (y2 > y1 and x2 > x1) else None

                if self.tracks[track_id] is None:
                    self.tracks[track_id] = TrackHistory(track_id)

                track_hist = self.tracks[track_id]
                should_check_face = track_hist.update([x1, y1, x2, y2], crop, self.interval)

                tracked_objects.append({
                    "track_id": track_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": round(conf, 3),
                    "crop": crop,
                    "trajectory": list(track_hist.trajectory),
                    "should_check_face": should_check_face
                })

        return tracked_objects

    def get_metrics(self):
        return self.profiler.get_summary()
