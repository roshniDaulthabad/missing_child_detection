from typing import List, Dict
from app.database.models import db, PotentialMatchAlert, Camera, MissingChild

class MovementService:
    """Aggregates cross-camera sightings and movements into structured timeline and map coordinates."""

    @staticmethod
    def get_movement_history(case_id: str) -> List[Dict]:
        alerts = PotentialMatchAlert.query.filter_by(case_id=case_id).order_by(
            PotentialMatchAlert.created_at.asc()
        ).all()

        movements = []
        for alt in alerts:
            cam = Camera.query.filter_by(camera_id=alt.camera_id).first()
            lat = cam.latitude if cam else 28.6139  # Default reference coordinates
            lon = cam.longitude if cam else 77.2090
            movements.append({
                "alert_id": alt.alert_id,
                "camera_id": alt.camera_id,
                "camera_location": alt.camera_location,
                "timestamp": alt.created_at.strftime("%Y-%m-%d %H:%M:%S") if alt.created_at else "Just now",
                "similarity": round(alt.similarity_score * 100, 1),
                "track_id": alt.track_id,
                "frame_path": alt.cctv_frame_path,
                "face_path": alt.face_crop_path,
                "status": alt.status,
                "latitude": lat,
                "longitude": lon
            })
        return movements

movement_service = MovementService()
