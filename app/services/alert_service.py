import time
from app.database.models import db, PotentialMatchAlert, MissingChild, CaseStatusHistory, AuditLog
from app.services.case_service import case_service

class AlertService:
    """Manages potential match alerts and officer verification workflows."""

    @staticmethod
    def create_alert(alert_data: dict) -> PotentialMatchAlert:
        alert = PotentialMatchAlert(
            alert_id=alert_data["alert_id"],
            case_id=alert_data["case_id"],
            camera_id=alert_data["camera_id"],
            camera_location=alert_data.get("camera_location", "CCTV Stream"),
            track_id=alert_data["track_id"],
            similarity_score=float(alert_data["similarity_score"]),
            cctv_frame_path=alert_data["cctv_frame_path"],
            face_crop_path=alert_data["face_crop_path"],
            reference_photo_path=alert_data.get("reference_photo_path", ""),
            model_version=alert_data.get("model_version", "adaface-iresnet-cpu-v1"),
            status="New"
        )
        db.session.add(alert)
        
        # Log audit trail
        audit = AuditLog(
            actor="AI_DETECTION_PIPELINE",
            action="ALERT_GENERATED",
            resource_type="PotentialMatchAlert",
            resource_id=alert.alert_id,
            details=f"Potential match detected for Case {alert.case_id} on {alert.camera_id} with score {alert.similarity_score:.3f}"
        )
        db.session.add(audit)
        db.session.commit()
        return alert

    @staticmethod
    def review_alert(alert_id: str, action: str, officer_name: str, notes: str = "") -> PotentialMatchAlert:
        """Officer verification action handler.
        Supported actions: 'confirm', 'reject', 'investigate', 'resolve'
        """
        alert = PotentialMatchAlert.query.filter_by(alert_id=alert_id).first()
        if not alert:
            raise ValueError(f"Alert {alert_id} not found")

        action = action.lower()
        if action == "confirm":
            alert.status = "Confirmed by Officer"
            # Update associated case to Potential Match / Active Investigation
            case_service.update_case_status(
                alert.case_id, "Potential Match", officer_name,
                f"Officer confirmed AI match from {alert.camera_id} (Score: {alert.similarity_score*100:.1f}%). Notes: {notes}"
            )
        elif action == "reject":
            alert.status = "Rejected"
        elif action == "investigate":
            alert.status = "Investigating"
            case_service.update_case_status(
                alert.case_id, "Active Investigation", officer_name,
                f"Investigation initiated based on sighting at {alert.camera_id}. Notes: {notes}"
            )
        elif action == "resolve":
            alert.status = "Resolved"
        else:
            alert.status = "Under Review"

        alert.reviewed_by = officer_name
        alert.reviewed_at = db.func.now()
        alert.review_notes = notes

        audit = AuditLog(
            actor=officer_name,
            action=f"ALERT_REVIEW_{action.upper()}",
            resource_type="PotentialMatchAlert",
            resource_id=alert_id,
            details=f"Officer {officer_name} executed '{action}' on Alert {alert_id}. Notes: {notes}"
        )
        db.session.add(audit)
        db.session.commit()
        return alert

alert_service = AlertService()
