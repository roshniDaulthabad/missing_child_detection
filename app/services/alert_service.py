import time
from app.database.models import db, PotentialMatchAlert, MissingChild, CaseStatusHistory, AuditLog
from app.services.case_service import case_service
from app.services.featherless_service import featherless_service

class AlertService:
    """Manages potential match alerts and officer verification workflows."""

    @staticmethod
    def create_alert(alert_data: dict) -> PotentialMatchAlert:
        # 1. Fetch child record for context-aware AI verification
        case_id = alert_data.get("case_id")
        child = MissingChild.query.filter_by(case_id=case_id).first() if case_id else None
        
        child_dict = {}
        if child:
            child_dict = {
                "case_id": child.case_id,
                "child_name": child.child_name,
                "age": child.age,
                "gender": child.gender,
                "date_missing": child.date_missing,
                "last_known_location": child.last_known_location,
                "physical_description": child.physical_description or "N/A",
                "identifying_characteristics": child.identifying_characteristics or "N/A",
                "priority": child.priority or "High"
            }

        # 2. Run Featherless.ai verification assessment
        ai_assessment = featherless_service.assess_potential_match(child_dict, alert_data)

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
            status="New",
            ai_status=ai_assessment.get("status", "Pending"),
            ai_confidence=ai_assessment.get("confidence", "N/A"),
            ai_assessment=ai_assessment.get("reasoning", ""),
            ai_recommendation=ai_assessment.get("recommendation", "")
        )
        db.session.add(alert)
        
        # Log audit trail with Featherless verification status
        audit = AuditLog(
            actor="FEATHERLESS_AI_VERIFIER",
            action="ALERT_GENERATED",
            resource_type="PotentialMatchAlert",
            resource_id=alert.alert_id,
            details=f"Alert generated for Case {alert.case_id} on {alert.camera_id} (Score: {alert.similarity_score:.3f}). Featherless AI Status: {alert.ai_status} ({alert.ai_confidence} Confidence)."
        )
        db.session.add(audit)
        db.session.commit()
        return alert

    @staticmethod
    def verify_alert_with_ai(alert_id: str) -> PotentialMatchAlert:
        """Runs or re-runs Featherless.ai verification on an existing alert."""
        alert = PotentialMatchAlert.query.filter_by(alert_id=alert_id).first()
        if not alert:
            raise ValueError(f"Alert {alert_id} not found")

        child = MissingChild.query.filter_by(case_id=alert.case_id).first()
        child_dict = {}
        if child:
            child_dict = {
                "case_id": child.case_id,
                "child_name": child.child_name,
                "age": child.age,
                "gender": child.gender,
                "date_missing": child.date_missing,
                "last_known_location": child.last_known_location,
                "physical_description": child.physical_description or "N/A",
                "identifying_characteristics": child.identifying_characteristics or "N/A",
                "priority": child.priority or "High"
            }

        alert_data = {
            "alert_id": alert.alert_id,
            "case_id": alert.case_id,
            "camera_id": alert.camera_id,
            "camera_location": alert.camera_location,
            "track_id": alert.track_id,
            "similarity_score": alert.similarity_score,
            "model_version": alert.model_version,
            "timestamp": alert.created_at.strftime("%Y-%m-%d %H:%M:%S") if alert.created_at else "Recent"
        }

        ai_assessment = featherless_service.assess_potential_match(child_dict, alert_data)
        alert.ai_status = ai_assessment.get("status", alert.ai_status)
        alert.ai_confidence = ai_assessment.get("confidence", alert.ai_confidence)
        alert.ai_assessment = ai_assessment.get("reasoning", alert.ai_assessment)
        alert.ai_recommendation = ai_assessment.get("recommendation", alert.ai_recommendation)

        audit = AuditLog(
            actor="FEATHERLESS_AI_VERIFIER",
            action="AI_REVERIFICATION",
            resource_type="PotentialMatchAlert",
            resource_id=alert.alert_id,
            details=f"Featherless AI re-evaluated Alert {alert.alert_id}: {alert.ai_status} ({alert.ai_confidence} Confidence)."
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
