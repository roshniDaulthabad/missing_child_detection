import time
from app.database.models import db, MissingChild, RecoveryRecord, WelfareRecord, CaseStatusHistory, AuditLog
from app.services.case_service import case_service

class WelfareService:
    """Manages post-location recovery protocols, guardian verification, and welfare referrals."""

    @staticmethod
    def record_recovery_and_welfare(case_id: str, data: dict, officer_name: str):
        child = MissingChild.query.filter_by(case_id=case_id).first()
        if not child:
            raise ValueError(f"Case {case_id} not found")

        # 1. Recovery Record
        recovery = RecoveryRecord(
            case_id=case_id,
            recovery_date=data.get("recovery_date", time.strftime("%Y-%m-%d")),
            recovery_location=data.get("recovery_location", "Safe Police Custody"),
            recovering_officer=officer_name,
            guardian_verified=bool(data.get("guardian_verified", False)),
            guardian_notes=data.get("guardian_notes", ""),
            current_status="Recovered"
        )
        db.session.add(recovery)

        # 2. Welfare Record
        welfare = WelfareRecord(
            case_id=case_id,
            medical_referral=bool(data.get("medical_referral", False)),
            medical_notes=data.get("medical_notes", ""),
            psychological_support=bool(data.get("psychological_support", False)),
            psychological_notes=data.get("psychological_notes", ""),
            temporary_shelter=bool(data.get("temporary_shelter", False)),
            shelter_details=data.get("shelter_details", ""),
            reunification_status=data.get("reunification_status", "Reunification Pending"),
            welfare_officer=officer_name,
            follow_up_notes=data.get("follow_up_notes", "")
        )
        db.session.add(welfare)

        # 3. Update case status based on reunification status
        final_status = "Reunified" if data.get("reunification_status") == "Reunified" else "Recovered"
        case_service.update_case_status(
            case_id, final_status, officer_name,
            f"Recovery documented. Guardian verified: {recovery.guardian_verified}. Reunification: {welfare.reunification_status}"
        )

        db.session.commit()
        return recovery, welfare

welfare_service = WelfareService()
