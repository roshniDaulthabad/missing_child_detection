import os
import time
from flask import Blueprint, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from app.config import Config, BASE_DIR
from app.database.models import (
    db, MissingChild, PotentialMatchAlert, Camera, AuditLog,
    CaseStatusHistory, RecoveryRecord, WelfareRecord
)
from app.services.case_service import case_service
from app.services.alert_service import alert_service
from app.services.movement_service import movement_service
from app.services.welfare_service import welfare_service
from app.ai.pipeline import worker_pool
from app.ai.hardware import HardwareProfile
from app.ai.pipeline.video_forensics import forensic_analyzer
from flask import current_app

api_bp = Blueprint("api", __name__, url_prefix="/api")

@api_bp.route("/stats", methods=["GET"])
def get_dashboard_stats():
    total_cases = MissingChild.query.count()
    active_cases = MissingChild.query.filter(
        MissingChild.status.in_(["Missing", "Active Investigation", "Potential Match"])
    ).count()
    located_cases = MissingChild.query.filter(
        MissingChild.status.in_(["Located", "Recovered", "Reunified"])
    ).count()
    closed_cases = MissingChild.query.filter_by(status="Closed").count()

    total_alerts = PotentialMatchAlert.query.count()
    pending_alerts = PotentialMatchAlert.query.filter_by(status="New").count()
    reviewed_alerts = PotentialMatchAlert.query.filter(
        PotentialMatchAlert.status.in_(["Confirmed by Officer", "Rejected", "Investigating", "Resolved"])
    ).count()

    cameras = Camera.query.all()
    active_cameras = [c for c in cameras if c.status == "online"]

    # Ingest runtime telemetry
    runtime = HardwareProfile.get_runtime_metrics()
    
    # Active worker metrics
    worker = worker_pool.workers.get("CAM-01")
    worker_stat = worker.get_status() if worker else {}

    return jsonify({
        "cases": {
            "total": total_cases,
            "active": active_cases,
            "located": located_cases,
            "closed": closed_cases
        },
        "alerts": {
            "total": total_alerts,
            "pending": pending_alerts,
            "reviewed": reviewed_alerts
        },
        "cameras": {
            "total": len(cameras),
            "online": len(active_cameras),
            "offline": len(cameras) - len(active_cameras)
        },
        "system_health": {
            "cpu_percent": runtime["cpu_percent"],
            "ram_percent": runtime["ram_percent"],
            "ram_used_gb": runtime["ram_used_gb"],
            "ram_total_gb": runtime["ram_total_gb"],
            "device": "CPU",
            "fps": worker_stat.get("fps", 0.0),
            "latency_ms": worker_stat.get("latency_ms", 0.0),
            "active_tracks": worker_stat.get("active_tracks", 0),
            "operation_mode": Config.OPERATION_MODE
        }
    })

@api_bp.route("/cases", methods=["GET", "POST"])
def handle_cases():
    if request.method == "POST":
        if "photo" not in request.files:
            return jsonify({"error": "Child photograph file is required"}), 400
        photo = request.files["photo"]
        if photo.filename == "":
            return jsonify({"error": "No selected photo file"}), 400

        data = request.form.to_dict()
        try:
            child = case_service.register_child(data, photo)
            return jsonify({
                "message": "Child registered successfully without model retraining",
                "case_id": child.case_id,
                "child_name": child.child_name,
                "status": child.status
            }), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # GET
    cases = MissingChild.query.order_by(MissingChild.created_at.desc()).all()
    results = []
    for c in cases:
        results.append({
            "case_id": c.case_id,
            "child_name": c.child_name,
            "age": c.age,
            "gender": c.gender,
            "parent_name": c.parent_name,
            "parent_contact": c.parent_contact,
            "last_known_location": c.last_known_location,
            "date_missing": c.date_missing,
            "priority": c.priority,
            "status": c.status,
            "assigned_officer": c.assigned_officer,
            "photo_path": f"/api/files/photo/{c.case_id}"
        })
    return jsonify(results)

@api_bp.route("/cases/<case_id>", methods=["GET"])
def get_case_detail(case_id):
    child = MissingChild.query.filter_by(case_id=case_id).first()
    if not child:
        return jsonify({"error": "Case not found"}), 404

    history = CaseStatusHistory.query.filter_by(case_id=case_id).order_by(CaseStatusHistory.timestamp.asc()).all()
    alerts = PotentialMatchAlert.query.filter_by(case_id=case_id).all()

    return jsonify({
        "case_id": child.case_id,
        "child_name": child.child_name,
        "age": child.age,
        "gender": child.gender,
        "parent_name": child.parent_name,
        "parent_contact": child.parent_contact,
        "date_missing": child.date_missing,
        "last_known_location": child.last_known_location,
        "physical_description": child.physical_description,
        "identifying_characteristics": child.identifying_characteristics,
        "priority": child.priority,
        "status": child.status,
        "assigned_officer": child.assigned_officer,
        "additional_notes": child.additional_notes,
        "photo_path": f"/api/files/photo/{child.case_id}",
        "photo_sha3": child.photo_hash,
        "history": [{
            "old_status": h.old_status,
            "new_status": h.new_status,
            "updated_by": h.updated_by,
            "notes": h.notes,
            "timestamp": h.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        } for h in history],
        "alerts_count": len(alerts)
    })

@api_bp.route("/cases/<case_id>/status", methods=["POST"])
def update_status(case_id):
    payload = request.get_json() or {}
    new_status = payload.get("status")
    officer = payload.get("officer", "Authorized Officer")
    notes = payload.get("notes", "")

    if not new_status:
        return jsonify({"error": "New status is required"}), 400

    try:
        child = case_service.update_case_status(case_id, new_status, officer, notes)
        return jsonify({"message": "Case status updated", "new_status": child.status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/alerts", methods=["GET"])
def get_alerts():
    alerts = PotentialMatchAlert.query.order_by(PotentialMatchAlert.created_at.desc()).all()
    out = []
    for a in alerts:
        out.append({
            "alert_id": a.alert_id,
            "case_id": a.case_id,
            "child_name": a.case.child_name if a.case else a.case_id,
            "camera_id": a.camera_id,
            "camera_location": a.camera_location,
            "track_id": a.track_id,
            "similarity_score": round(a.similarity_score * 100, 1),
            "status": a.status,
            "model_version": a.model_version,
            "frame_url": f"/api/files/frame/{a.alert_id}",
            "face_url": f"/api/files/face/{a.alert_id}",
            "reference_url": f"/api/files/photo/{a.case_id}",
            "reviewed_by": a.reviewed_by,
            "reviewed_at": a.reviewed_at.strftime("%Y-%m-%d %H:%M:%S") if a.reviewed_at else None,
            "review_notes": a.review_notes,
            "timestamp": a.created_at.strftime("%Y-%m-%d %H:%M:%S") if a.created_at else None
        })
    return jsonify(out)

@api_bp.route("/alerts/<alert_id>/review", methods=["POST"])
def review_alert_endpoint(alert_id):
    payload = request.get_json() or {}
    action = payload.get("action", "confirm")
    officer = payload.get("officer", "Authorized Officer")
    notes = payload.get("notes", "")

    try:
        alert = alert_service.review_alert(alert_id, action, officer, notes)
        return jsonify({
            "message": f"Alert reviewed: {action}",
            "alert_id": alert.alert_id,
            "new_status": alert.status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/stream/start", methods=["POST"])
def start_stream():
    payload = request.get_json() or {}
    camera_id = payload.get("camera_id", "CAM-01")
    source = payload.get("source", "demo")
    location = payload.get("location", "Main Gate")

    worker = worker_pool.get_or_create_worker(camera_id, source, location)
    
    # Connect alert callback to save directly to DB
    def on_alert(alert_dict):
        # Run inside application context
        from app import app
        with app.app_context():
            alert_service.create_alert(alert_dict)

    worker.on_alert_callback = on_alert
    worker.start()
    return jsonify({"message": f"Stream worker started for {camera_id}", "status": "running"})

@api_bp.route("/stream/stop", methods=["POST"])
def stop_stream():
    payload = request.get_json() or {}
    camera_id = payload.get("camera_id", "CAM-01")
    worker_pool.stop_worker(camera_id)
    return jsonify({"message": f"Stream worker stopped for {camera_id}"})

@api_bp.route("/stream/status", methods=["GET"])
def get_stream_status():
    camera_id = request.args.get("camera_id", "CAM-01")
    worker = worker_pool.workers.get(camera_id)
    if not worker:
        return jsonify({"is_running": False, "camera_id": camera_id})
    return jsonify(worker.get_status())

@api_bp.route("/stream/video_feed/<camera_id>", methods=["GET"])
def video_feed(camera_id):
    worker = worker_pool.workers.get(camera_id)
    if not worker:
        worker = worker_pool.get_or_create_worker(camera_id, "demo")
        worker.start()

    def generate():
        while True:
            frame_bytes = worker.get_latest_frame_bytes()
            if frame_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.04)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@api_bp.route("/movement/<case_id>", methods=["GET"])
def get_movement(case_id):
    movements = movement_service.get_movement_history(case_id)
    return jsonify(movements)

@api_bp.route("/welfare/<case_id>", methods=["POST"])
def handle_welfare(case_id):
    payload = request.get_json() or {}
    officer = payload.get("officer", "Welfare Officer")
    try:
        rec, wel = welfare_service.record_recovery_and_welfare(case_id, payload, officer)
        return jsonify({"message": "Recovery and welfare evaluation saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/cameras", methods=["GET", "POST"])
def handle_cameras():
    if request.method == "POST":
        data = request.get_json() or {}
        cam = Camera(
            camera_id=data.get("camera_id"),
            name=data.get("name"),
            location=data.get("location"),
            latitude=float(data.get("latitude", 28.6139)),
            longitude=float(data.get("longitude", 77.2090)),
            status="online",
            fps=float(data.get("fps", 15.0))
        )
        db.session.add(cam)
        db.session.commit()
        return jsonify({"message": "Camera added", "camera_id": cam.camera_id}), 201

    cams = Camera.query.all()
    return jsonify([{
        "camera_id": c.camera_id,
        "name": c.name,
        "location": c.location,
        "latitude": c.latitude,
        "longitude": c.longitude,
        "status": c.status,
        "fps": c.fps
    } for c in cams])

@api_bp.route("/audit", methods=["GET"])
def get_audit_logs():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(100).all()
    return jsonify([{
        "id": l.id,
        "actor": l.actor,
        "action": l.action,
        "resource_type": l.resource_type,
        "resource_id": l.resource_id,
        "details": l.details,
        "timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "integrity_hash": l.integrity_hash
    } for l in logs])

# File delivery endpoints
@api_bp.route("/files/photo/<case_id>", methods=["GET"])
def get_photo(case_id):
    placeholder = os.path.abspath(os.path.join(Config.BASE_DIR, "app", "static", "img", "placeholder.png"))
    child = MissingChild.query.filter_by(case_id=case_id).first()
    if child and child.photo_path:
        for candidate in [child.photo_path, os.path.abspath(child.photo_path), os.path.join(Config.BASE_DIR, child.photo_path)]:
            if os.path.exists(candidate):
                return send_file(os.path.abspath(candidate), mimetype="image/jpeg")
    return send_file(placeholder, mimetype="image/png")

@api_bp.route("/files/frame/<alert_id>", methods=["GET"])
def get_alert_frame(alert_id):
    placeholder = os.path.abspath(os.path.join(Config.BASE_DIR, "app", "static", "img", "placeholder.png"))
    alert = PotentialMatchAlert.query.filter_by(alert_id=alert_id).first()
    if alert and alert.cctv_frame_path:
        for candidate in [alert.cctv_frame_path, os.path.abspath(alert.cctv_frame_path), os.path.join(Config.BASE_DIR, alert.cctv_frame_path)]:
            if os.path.exists(candidate):
                return send_file(os.path.abspath(candidate), mimetype="image/jpeg")
    return send_file(placeholder, mimetype="image/png")

@api_bp.route("/files/face/<alert_id>", methods=["GET"])
def get_alert_face(alert_id):
    placeholder = os.path.abspath(os.path.join(Config.BASE_DIR, "app", "static", "img", "placeholder.png"))
    alert = PotentialMatchAlert.query.filter_by(alert_id=alert_id).first()
    if alert and alert.face_crop_path:
        for candidate in [alert.face_crop_path, os.path.abspath(alert.face_crop_path), os.path.join(Config.BASE_DIR, alert.face_crop_path)]:
            if os.path.exists(candidate):
                return send_file(os.path.abspath(candidate), mimetype="image/jpeg")
    return send_file(placeholder, mimetype="image/png")

# --- True Uploaded Video Forensic Analysis Endpoints ---

@api_bp.route("/footage/upload", methods=["POST"])
def upload_footage():
    """Receives an uploaded CCTV video file, initializes forensic state, and starts background analysis."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file or video_file.filename == "":
        return jsonify({"error": "No selected video file"}), 400

    # Validate file extension
    ext = os.path.splitext(video_file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mkv", ".mov", ".webm"]:
        return jsonify({"error": f"Unsupported video format '{ext}'. Supported: .mp4, .avi, .mkv, .mov, .webm"}), 400

    cam_id = request.form.get("camera_id", "CAM-FOOTAGE")
    location = request.form.get("location", "Forensic Upload")

    os.makedirs("data/uploads/footage", exist_ok=True)
    filename = secure_filename(f"{int(time.time())}_{video_file.filename}")
    save_path = os.path.join("data", "uploads", "footage", filename)
    video_file.save(save_path)

    # Launch forensic analyzer with app instance for thread-safe alert DB commits
    app_instance = current_app._get_current_object()
    job_id = forensic_analyzer.start_job(save_path, camera_id=cam_id, location=location, app=app_instance)

    return jsonify({
        "message": "Video uploaded successfully. Forensic AI pipeline initiated.",
        "job_id": job_id,
        "camera_id": cam_id,
        "location": location,
        "filename": filename
    }), 202

@api_bp.route("/footage/status/<job_id>", methods=["GET"])
def get_footage_status(job_id):
    job = forensic_analyzer.get_job(job_id)
    if not job:
        return jsonify({"error": f"Forensic job '{job_id}' not found"}), 404
    return jsonify(job.to_dict())

@api_bp.route("/footage/cancel/<job_id>", methods=["POST"])
def cancel_footage_job(job_id):
    success = forensic_analyzer.cancel_job(job_id)
    return jsonify({"success": success, "message": "Forensic job cancellation requested"})

@api_bp.route("/footage/preview/<job_id>", methods=["GET"])
def get_footage_preview(job_id):
    job = forensic_analyzer.get_job(job_id)
    if job and job.latest_frame_jpeg:
        return Response(job.latest_frame_jpeg, mimetype="image/jpeg")
    placeholder = os.path.abspath(os.path.join(Config.BASE_DIR, "app", "static", "img", "placeholder.png"))
    return send_file(placeholder, mimetype="image/png")

@api_bp.route("/footage/keyframe/<filename>", methods=["GET"])
def get_keyframe(filename):
    safe_name = secure_filename(filename)
    path = os.path.abspath(os.path.join(Config.BASE_DIR, "data", "keyframes", safe_name))
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    placeholder = os.path.abspath(os.path.join(Config.BASE_DIR, "app", "static", "img", "placeholder.png"))
    return send_file(placeholder, mimetype="image/png")

@api_bp.route("/footage/video/<job_id>", methods=["GET"])
def get_processed_video(job_id):
    job = forensic_analyzer.get_job(job_id)
    if job and job.output_video_path and os.path.exists(job.output_video_path):
        return send_file(os.path.abspath(job.output_video_path), mimetype="video/mp4")
    return jsonify({"error": "Processed output video not available"}), 404

