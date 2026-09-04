import time
from typing import List, Dict, Optional
import numpy as np
from app.config import Config
from app.ai.reid.feature_extractor import reid_extractor_service

class CameraTopologyGraph:
    """Represents camera spatial layout and realistic pedestrian transit transition times."""

    def __init__(self):
        # Directed transit constraints: (cam_from, cam_to) -> (min_sec, max_sec)
        self.transitions = {
            ("CAM-01", "CAM-02"): (5.0, 180.0),
            ("CAM-02", "CAM-01"): (5.0, 180.0),
            ("CAM-02", "CAM-03"): (10.0, 240.0),
            ("CAM-03", "CAM-02"): (10.0, 240.0),
            ("CAM-01", "CAM-03"): (15.0, 360.0),
            ("CAM-03", "CAM-01"): (15.0, 360.0),
            ("CAM-03", "CAM-04"): (10.0, 300.0),
            ("CAM-04", "CAM-03"): (10.0, 300.0),
        }

    def is_spatially_reachable(self, cam_from: str, cam_to: str) -> bool:
        if cam_from == cam_to:
            return True
        return (cam_from, cam_to) in self.transitions

    def is_temporally_valid(self, cam_from: str, cam_to: str, delta_sec: float) -> bool:
        if cam_from == cam_to:
            return True
        key = (cam_from, cam_to)
        if key not in self.transitions:
            # Default fallback for arbitrary camera networks: 5s to 15 minutes
            return 5.0 <= delta_sec <= 900.0
        min_t, max_t = self.transitions[key]
        return min_t <= delta_sec <= max_t


class CrossCameraAssociator:
    """Associates pedestrian tracklets across disparate CCTV cameras using spatio-temporal
    topology constraints, appearance Re-ID similarity, and biometric face evidence.
    """

    def __init__(self, sim_threshold: float = None):
        self.sim_threshold = sim_threshold or Config.REID_SIMILARITY_THRESHOLD
        self.topology = CameraTopologyGraph()
        # Tracklet registry: {f"{camera_id}_{track_id}": tracklet_info}
        self.tracklets: Dict[str, dict] = {}

    def register_tracklet(self, camera_id: str, track_id: int, appearance_feat: np.ndarray,
                          timestamp: float = None, face_match_case: str = None):
        key = f"{camera_id}_{track_id}"
        self.tracklets[key] = {
            "key": key,
            "camera_id": camera_id,
            "track_id": track_id,
            "appearance_feat": appearance_feat,
            "timestamp": timestamp or time.time(),
            "face_match_case": face_match_case
        }

    def find_associations(self, target_cam: str, target_track_id: int) -> List[dict]:
        """Finds candidate historical or sequential sightings matching the target track across other cameras."""
        target_key = f"{target_cam}_{target_track_id}"
        target = self.tracklets.get(target_key)
        if not target or target["appearance_feat"] is None:
            return []

        candidates = []
        for key, other in self.tracklets.items():
            if other["camera_id"] == target["camera_id"]:
                continue  # Skip same camera

            delta_t = abs(target["timestamp"] - other["timestamp"])
            # 1. Spatio-temporal feasibility check
            if not self.topology.is_temporally_valid(other["camera_id"], target["camera_id"], delta_t):
                continue

            # 2. Appearance Re-ID similarity
            app_sim = reid_extractor_service.appearance_similarity(
                target["appearance_feat"], other["appearance_feat"]
            )

            # 3. Biometric face evidence bonus
            face_evidence_boost = 0.0
            if target["face_match_case"] and other["face_match_case"]:
                if target["face_match_case"] == other["face_match_case"]:
                    face_evidence_boost = 0.35  # Strong joint evidence

            composite_confidence = round(min(1.0, app_sim * 0.7 + face_evidence_boost), 3)

            if composite_confidence >= self.sim_threshold:
                candidates.append({
                    "from_camera": other["camera_id"],
                    "from_track_id": other["track_id"],
                    "to_camera": target["camera_id"],
                    "to_track_id": target["track_id"],
                    "time_delta_sec": round(delta_t, 1),
                    "appearance_similarity": app_sim,
                    "confidence": composite_confidence,
                    "face_evidence": target["face_match_case"] or other["face_match_case"],
                    "status": "Candidate Cross-Camera Association"
                })

        return sorted(candidates, key=lambda x: x["confidence"], reverse=True)

cross_camera_service = CrossCameraAssociator()
