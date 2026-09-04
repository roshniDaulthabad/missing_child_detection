from datetime import datetime, timezone
import json
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default="officer")  # admin, investigator, officer
    full_name = db.Column(db.String(120), nullable=False)
    badge_number = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class MissingChild(db.Model):
    __tablename__ = "missing_children"
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    child_name = db.Column(db.String(120), nullable=False, index=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    photo_path = db.Column(db.String(255), nullable=False)
    photo_hash = db.Column(db.String(64), nullable=True)  # SHA-3-256 integrity hash
    parent_name = db.Column(db.String(120), nullable=False)
    parent_contact = db.Column(db.String(50), nullable=False)
    date_missing = db.Column(db.String(50), nullable=False)
    time_missing = db.Column(db.String(50), nullable=True)
    last_known_location = db.Column(db.String(255), nullable=False)
    physical_description = db.Column(db.Text, nullable=True)
    identifying_characteristics = db.Column(db.Text, nullable=True)
    priority = db.Column(db.String(20), default="High")  # Critical, High, Medium
    status = db.Column(db.String(50), default="Missing", index=True)
    assigned_officer = db.Column(db.String(120), nullable=True)
    additional_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    embeddings = db.relationship("ChildEmbedding", backref="case", cascade="all, delete-orphan")
    alerts = db.relationship("PotentialMatchAlert", backref="case", cascade="all, delete-orphan")
    status_history = db.relationship("CaseStatusHistory", backref="case", cascade="all, delete-orphan")


class ChildEmbedding(db.Model):
    __tablename__ = "child_embeddings"
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.String(64), db.ForeignKey("missing_children.case_id"), nullable=False, index=True)
    embedding_json = db.Column(db.Text, nullable=False)  # 512-D float list serialized
    reference_type = db.Column(db.String(50), default="original")  # original or age_progressed
    model_version = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def get_vector(self):
        return json.loads(self.embedding_json)


class Camera(db.Model):
    __tablename__ = "cameras"
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    location = db.Column(db.String(255), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    rtsp_url_encrypted = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default="online")  # online, offline, degraded
    fps = db.Column(db.Float, default=15.0)
    enabled = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class Track(db.Model):
    __tablename__ = "tracks"
    id = db.Column(db.Integer, primary_key=True)
    track_id = db.Column(db.Integer, nullable=False, index=True)
    camera_id = db.Column(db.String(50), nullable=False, index=True)
    first_seen = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    active = db.Column(db.Boolean, default=True)
    reid_embedding_json = db.Column(db.Text, nullable=True)


class TrackObservation(db.Model):
    __tablename__ = "track_observations"
    id = db.Column(db.Integer, primary_key=True)
    track_id = db.Column(db.Integer, nullable=False, index=True)
    camera_id = db.Column(db.String(50), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    bbox_json = db.Column(db.String(255), nullable=False)
    face_quality_score = db.Column(db.Float, nullable=True)
    face_detected = db.Column(db.Boolean, default=False)
    face_crop_path = db.Column(db.String(255), nullable=True)


class CrossCameraMatch(db.Model):
    __tablename__ = "cross_camera_matches"
    id = db.Column(db.Integer, primary_key=True)
    primary_track_id = db.Column(db.Integer, nullable=False)
    candidate_track_id = db.Column(db.Integer, nullable=False)
    camera_from = db.Column(db.String(50), nullable=False)
    camera_to = db.Column(db.String(50), nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    time_delta_sec = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(30), default="candidate")  # candidate, verified, rejected
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class PotentialMatchAlert(db.Model):
    __tablename__ = "alerts"
    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    case_id = db.Column(db.String(64), db.ForeignKey("missing_children.case_id"), nullable=False, index=True)
    camera_id = db.Column(db.String(50), nullable=False)
    camera_location = db.Column(db.String(255), nullable=False)
    track_id = db.Column(db.Integer, nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    cctv_frame_path = db.Column(db.String(255), nullable=False)
    face_crop_path = db.Column(db.String(255), nullable=False)
    reference_photo_path = db.Column(db.String(255), nullable=False)
    model_version = db.Column(db.String(80), nullable=False)
    status = db.Column(db.String(50), default="New", index=True)  # New, Under Review, Confirmed by Officer, Rejected, Investigating, Resolved
    review_notes = db.Column(db.Text, nullable=True)
    reviewed_by = db.Column(db.String(120), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class CaseStatusHistory(db.Model):
    __tablename__ = "case_status_history"
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.String(64), db.ForeignKey("missing_children.case_id"), nullable=False, index=True)
    old_status = db.Column(db.String(50), nullable=False)
    new_status = db.Column(db.String(50), nullable=False)
    updated_by = db.Column(db.String(120), nullable=False)
    notes = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class RecoveryRecord(db.Model):
    __tablename__ = "recovery_records"
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.String(64), db.ForeignKey("missing_children.case_id"), nullable=False, index=True)
    recovery_date = db.Column(db.String(50), nullable=False)
    recovery_location = db.Column(db.String(255), nullable=False)
    recovering_officer = db.Column(db.String(120), nullable=False)
    guardian_verified = db.Column(db.Boolean, default=False)
    guardian_notes = db.Column(db.Text, nullable=True)
    current_status = db.Column(db.String(50), default="Recovered")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class WelfareRecord(db.Model):
    __tablename__ = "welfare_records"
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.String(64), db.ForeignKey("missing_children.case_id"), nullable=False, index=True)
    medical_referral = db.Column(db.Boolean, default=False)
    medical_notes = db.Column(db.Text, nullable=True)
    psychological_support = db.Column(db.Boolean, default=False)
    psychological_notes = db.Column(db.Text, nullable=True)
    temporary_shelter = db.Column(db.Boolean, default=False)
    shelter_details = db.Column(db.Text, nullable=True)
    reunification_status = db.Column(db.String(50), default="Pending")
    welfare_officer = db.Column(db.String(120), nullable=False)
    follow_up_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True)
    actor = db.Column(db.String(120), nullable=False)
    action = db.Column(db.String(120), nullable=False)
    resource_type = db.Column(db.String(80), nullable=False)
    resource_id = db.Column(db.String(80), nullable=True)
    details = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    integrity_hash = db.Column(db.String(64), nullable=True)
