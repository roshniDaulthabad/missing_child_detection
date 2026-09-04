import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename
from app.config import Config
from app.database.models import db, MissingChild, ChildEmbedding, CaseStatusHistory, AuditLog
from app.security import crypto_service, integrity_service
from app.ai.face import face_detector_service, face_quality_service, face_recognizer_service
from app.ai.age_progression.age_model import age_progression_service

class CaseService:
    """Manages missing child cases, one-time face embedding extraction, and secure storage."""

    @staticmethod
    def register_child(data: dict, photo_file) -> MissingChild:
        case_id = data.get("case_id") or f"MC-2026-{int(time.time()) % 10000:04d}"
        
        # 1. Save uploaded photo
        filename = secure_filename(photo_file.filename or f"{case_id}.jpg")
        os.makedirs(Config.UPLOADS_DIR, exist_ok=True)
        os.makedirs(Config.ENCRYPTED_DIR, exist_ok=True)
        upload_path = str(Config.UPLOADS_DIR / f"{case_id}_{filename}")
        photo_file.save(upload_path)

        # 2. Compute SHA-3-256 integrity hash
        sha3_hash = integrity_service.hash_file(upload_path)

        # 3. Encrypt file with AES-256-GCM for secure cold storage
        encrypted_path = str(Config.ENCRYPTED_DIR / f"{case_id}_{filename}.enc")
        crypto_service.encrypt_file(upload_path, encrypted_path)

        # 4. Extract face crop and 512-D embedding ONCE
        img_bgr = cv2.imread(upload_path)
        if img_bgr is None:
            raise ValueError(f"Could not decode image file: {filename}")

        # Quality check & detection
        face_dets = face_detector_service.detect_faces(img_bgr)
        if not face_dets:
            # If no face detector hit, use whole crop for enrollment
            face_crop = img_bgr
        else:
            face_crop = face_dets[0]["crop"]

        quality_res = face_quality_service.assess_crop(face_crop)
        primary_embedding = face_recognizer_service.generate_embedding(face_crop)
        if primary_embedding is None:
            raise ValueError("Failed to generate facial embedding from uploaded photo.")

        # 5. Create MissingChild DB Record
        child = MissingChild(
            case_id=case_id,
            child_name=data.get("child_name", "Unknown"),
            age=int(data.get("age", 10)),
            gender=data.get("gender", "Unknown"),
            photo_path=upload_path,
            photo_hash=sha3_hash,
            parent_name=data.get("parent_name", "Unknown"),
            parent_contact=data.get("parent_contact", "N/A"),
            date_missing=data.get("date_missing", time.strftime("%Y-%m-%d")),
            time_missing=data.get("time_missing", time.strftime("%H:%M")),
            last_known_location=data.get("last_known_location", "Unknown Location"),
            physical_description=data.get("physical_description", ""),
            identifying_characteristics=data.get("identifying_characteristics", ""),
            priority=data.get("priority", "High"),
            status="Missing",
            assigned_officer=data.get("assigned_officer", "Duty Officer"),
            additional_notes=data.get("additional_notes", "")
        )
        db.session.add(child)

        # 6. Store Primary Embedding in DB
        emb_record = ChildEmbedding(
            case_id=case_id,
            embedding_json=json.dumps(primary_embedding.tolist()),
            reference_type="original",
            model_version=Config.FACE_MODEL_VERSION
        )
        db.session.add(emb_record)

        # 7. Generate Auxiliary Age Progression Reference Hypotheses
        try:
            aux_refs = age_progression_service.generate_reference_set(face_crop, child.age)
            for ref in aux_refs:
                aux_emb = face_recognizer_service.generate_embedding(ref["image"])
                if aux_emb is not None:
                    aux_record = ChildEmbedding(
                        case_id=case_id,
                        embedding_json=json.dumps(aux_emb.tolist()),
                        reference_type=f"age_progressed_+{ref['delta_years']}y",
                        model_version=ref["model_version"]
                    )
                    db.session.add(aux_record)
        except Exception as e:
            print(f"[WARNING] Age progression hypothesis generation note: {e}")

        # 8. Record Initial Timeline & Audit Log
        history = CaseStatusHistory(
            case_id=case_id,
            old_status="New Case Registered",
            new_status="Missing",
            updated_by=child.assigned_officer or "System",
            notes="Initial child registration, AES-256 encryption, and 512-D embedding enrollment."
        )
        audit = AuditLog(
            actor=child.assigned_officer or "System",
            action="CASE_REGISTERED",
            resource_type="MissingChild",
            resource_id=case_id,
            details=f"Enrolled child {child.child_name}, Age {child.age}. SHA3: {sha3_hash[:16]}...",
            integrity_hash=sha3_hash
        )
        db.session.add(history)
        db.session.add(audit)
        db.session.commit()

        # 9. Cache in memory for real-time vector search
        face_recognizer_service.register_case_embedding(
            case_id, primary_embedding,
            {"name": child.child_name, "age": child.age, "photo_path": upload_path}
        )
        print(f"[CASE SERVICE] Registered {case_id} ({child.child_name}). Cached embedding in memory.")
        return child

    @staticmethod
    def load_active_embeddings_to_cache():
        """Loads all active missing children embeddings from database into memory cache on startup."""
        active_children = MissingChild.query.filter(
            MissingChild.status.in_(["Missing", "Active Investigation", "Potential Match"])
        ).all()
        
        count = 0
        for ch in active_children:
            for emb in ch.embeddings:
                if emb.reference_type == "original":
                    vec = np.array(emb.get_vector(), dtype=np.float32)
                    face_recognizer_service.register_case_embedding(
                        ch.case_id, vec,
                        {"name": ch.child_name, "age": ch.age, "photo_path": ch.photo_path}
                    )
                    count += 1
        print(f"[CASE SERVICE] Loaded {count} active embeddings into memory cache.")
        return count

    @staticmethod
    def update_case_status(case_id: str, new_status: str, updated_by: str, notes: str = "") -> MissingChild:
        child = MissingChild.query.filter_by(case_id=case_id).first()
        if not child:
            raise ValueError(f"Case {case_id} not found")
            
        old_status = child.status
        child.status = new_status
        
        history = CaseStatusHistory(
            case_id=case_id,
            old_status=old_status,
            new_status=new_status,
            updated_by=updated_by,
            notes=notes
        )
        audit = AuditLog(
            actor=updated_by,
            action="STATUS_UPDATED",
            resource_type="MissingChild",
            resource_id=case_id,
            details=f"Status changed from '{old_status}' to '{new_status}'. Notes: {notes}"
        )
        db.session.add(history)
        db.session.add(audit)
        db.session.commit()

        if new_status in ["Recovered", "Reunified", "Closed"]:
            face_recognizer_service.remove_case_embedding(case_id)

        return child

case_service = CaseService()
