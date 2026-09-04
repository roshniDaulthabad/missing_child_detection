import os
import cv2
import time
import uuid
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from app.config import Config
from app.ai.tracking.tracker import PersonTracker
from app.ai.face import face_detector_service, face_quality_service, face_recognizer_service
from app.ai.enhancement.enhancer import face_enhancer_service
from app.ai.reid.feature_extractor import reid_extractor_service
from app.ai.reid.association import cross_camera_service
from app.services.alert_service import alert_service

class ForensicJobState:
    def __init__(self, job_id: str, video_path: str, camera_id: str, location: str):
        self.job_id = job_id
        self.video_path = video_path
        self.camera_id = camera_id
        self.location = location
        self.status = "queued"  # queued, processing, completed, cancelled, error
        self.error_message: Optional[str] = None
        
        self.total_frames = 0
        self.current_frame = 0
        self.progress_percent = 0.0
        self.video_fps = 30.0
        self.duration_seconds = 0.0
        
        self.processing_fps = 0.0
        self.latency_ms = 0.0
        self.unique_tracks = set()
        self.detected_persons_count = 0
        
        self.potential_matches: List[dict] = []
        self.keyframes: List[dict] = []
        self.output_video_path: Optional[str] = None
        self.latest_frame_jpeg: Optional[bytes] = None
        self.start_time = 0.0
        self.end_time = 0.0
        self.cancel_requested = False

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "camera_id": self.camera_id,
            "location": self.location,
            "status": self.status,
            "error_message": self.error_message,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame,
            "progress_percent": round(self.progress_percent, 1),
            "video_fps": round(self.video_fps, 1),
            "duration_seconds": round(self.duration_seconds, 1),
            "processing_fps": round(self.processing_fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "unique_tracks_count": len(self.unique_tracks),
            "detected_persons_count": self.detected_persons_count,
            "potential_matches_count": len(self.potential_matches),
            "potential_matches": self.potential_matches,
            "keyframes_count": len(self.keyframes),
            "keyframes": self.keyframes[-30:],  # Return recent keyframes
            "has_output_video": bool(self.output_video_path and os.path.exists(self.output_video_path)),
            "elapsed_seconds": round((time.time() - self.start_time) if self.start_time else 0.0, 1)
        }


class ForensicVideoAnalyzer:
    """True Uploaded Video Forensic Analysis Engine running strictly on CPU.
    Performs frame-by-frame person detection, multi-object tracking, face recognition
    against registered missing children, keyframe recording, and processed video generation.
    """

    def __init__(self):
        self.jobs: Dict[str, ForensicJobState] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

    def start_job(self, video_path: str, camera_id: str = "CAM-FOOTAGE", location: str = "Forensic Upload", app=None, app_context=None) -> str:
        job_id = f"JOB-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        job = ForensicJobState(job_id, video_path, camera_id, location)
        
        target_app = app
        if target_app is None and app_context is not None:
            target_app = getattr(app_context, 'app', None)

        with self.lock:
            self.jobs[job_id] = job
            
        thread = threading.Thread(
            target=self._run_analysis_worker,
            args=(job, target_app),
            daemon=True
        )
        self.threads[job_id] = thread
        thread.start()
        print(f"[FORENSIC ANALYZER] Started job {job_id} for file: {video_path}")
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].cancel_requested = True
                print(f"[FORENSIC ANALYZER] Cancel requested for job {job_id}")
                return True
        return False

    def get_job(self, job_id: str) -> Optional[ForensicJobState]:
        with self.lock:
            return self.jobs.get(job_id)

    def _format_timestamp(self, seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"

    def _run_analysis_worker(self, job: ForensicJobState, app=None):
        job.status = "processing"
        job.start_time = time.time()

        if not os.path.exists(job.video_path):
            job.status = "error"
            job.error_message = f"Uploaded video file not found at: {job.video_path}"
            return

        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            job.status = "error"
            job.error_message = f"Failed to open video container: {job.video_path}"
            return

        job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        job.video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        job.duration_seconds = (job.total_frames / job.video_fps) if job.video_fps > 0 else 0.0

        if job.total_frames <= 0:
            # Fallback estimation if container lacks metadata
            job.total_frames = 100

        # Initialize VideoWriter for processed output
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/keyframes", exist_ok=True)
        os.makedirs("data/snapshots", exist_ok=True)
        
        output_filename = f"processed_{job.job_id}.mp4"
        output_path = str(Path("data/processed") / output_filename)
        
        # Try mp4v codec for standard compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, min(job.video_fps, 25.0), (width, height))

        # Instantiate tracking engine strictly on CPU
        tracker = PersonTracker(model_path=Config.YOLO_MODEL_PATH)
        tracker.conf = 0.30  # High sensitivity for surveillance / CCTV frames

        frame_count = 0
        latencies = []
        alerted_case_ids = set()
        alert_records_by_case = {}
        best_similarity_by_case = {}

        # Check database logs: sync all reported missing children cases into biometric memory cache
        if app is not None:
            try:
                with app.app_context():
                    from app.services.case_service import case_service
                    loaded = case_service.load_active_embeddings_to_cache()
                    print(f"[FORENSIC ANALYZER] Checked database logs: {loaded} active reported cases loaded for matching.")
            except Exception as e:
                print(f"[FORENSIC ANALYZER] Warning: Failed to reload cases from database: {e}")

        try:
            while cap.isOpened():
                if job.cancel_requested:
                    job.status = "cancelled"
                    print(f"[FORENSIC ANALYZER] Job {job.job_id} cancelled by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                job.current_frame = frame_count
                t0 = time.time()
                
                # Video timestamp calculation
                current_time_sec = frame_count / job.video_fps
                time_str = self._format_timestamp(current_time_sec)

                # 1. Multi-object tracking on CPU
                tracked = tracker.update(frame)
                annotated = frame.copy()
                job.detected_persons_count += len(tracked)

                # Overlay forensic OSD
                osd_top = f"FORENSIC ANALYSIS: {job.camera_id} | TIME: {time_str} | FRAME: {frame_count}/{job.total_frames}"
                cv2.rectangle(annotated, (0, 0), (width, 45), (15, 20, 30), -1)
                cv2.putText(annotated, osd_top, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                # 2. Inspect tracked persons
                for obj in tracked:
                    track_id = obj["track_id"]
                    job.unique_tracks.add(track_id)
                    x1, y1, x2, y2 = obj["bbox"]
                    crop = obj["crop"]
                    should_check_face = obj["should_check_face"]
                    trajectory = obj["trajectory"]

                    # Draw motion trajectory
                    for pt in trajectory[-15:]:
                        cv2.circle(annotated, pt, 2, (0, 255, 255), -1)

                    box_color = (0, 255, 0)
                    label = f"ID: {track_id}"
                    face_quality_desc = "Not Evaluated"
                    is_match = False
                    matched_case = None

                    # Periodic Face Check
                    if crop is not None and (should_check_face or frame_count % 15 == 0):
                        face_dets = face_detector_service.detect_faces(crop)
                        for fdet in face_dets:
                            fcrop = fdet["crop"]
                            q_res = face_quality_service.assess_crop(fcrop)
                            face_quality_desc = f"{q_res['score'] * 100:.0f}%"

                            # Selective enhancement
                            if face_enhancer_service.should_enhance(fcrop, q_res["score"]):
                                fcrop = face_enhancer_service.enhance_face(fcrop)
                                q_res = face_quality_service.assess_crop(fcrop)

                            if q_res["sufficient"]:
                                # 512-D embedding extraction on CPU
                                emb = face_recognizer_service.generate_embedding(fcrop)
                                # Vector similarity search against active missing cases in memory
                                match = face_recognizer_service.determine_candidate(emb)

                                if match:
                                    is_match = True
                                    matched_case = match
                                    box_color = (0, 0, 255)
                                    sim_pct = match['similarity'] * 100
                                    label = f"MATCH: {match['case_id']} ({sim_pct:.1f}%)"

                                    case_id = match["case_id"]
                                    sim = float(match["similarity"])

                                    # Only ask for officer confirmation ONCE per child in this video analysis.
                                    if case_id not in alerted_case_ids:
                                        alerted_case_ids.add(case_id)
                                        best_similarity_by_case[case_id] = sim

                                        alert_uuid = f"ALT-VID-{uuid.uuid4().hex[:12]}"
                                        frame_file = os.path.abspath(f"data/snapshots/frame_{alert_uuid}.jpg")
                                        face_file = os.path.abspath(f"data/snapshots/face_{alert_uuid}.jpg")
                                        
                                        cv2.imwrite(frame_file, frame)
                                        cv2.imwrite(face_file, fcrop)

                                        alert_data = {
                                            "alert_id": alert_uuid,
                                            "case_id": case_id,
                                            "camera_id": job.camera_id,
                                            "camera_location": f"{job.location} [Video Time: {time_str}]",
                                            "track_id": track_id,
                                            "similarity_score": sim,
                                            "cctv_frame_path": frame_file,
                                            "face_crop_path": face_file,
                                            "reference_photo_path": match["metadata"].get("photo_path", ""),
                                            "model_version": Config.FACE_MODEL_VERSION,
                                            "status": "New",
                                            "timestamp": time_str
                                        }
                                        alert_records_by_case[case_id] = alert_data
                                        job.potential_matches.append(alert_data)

                                        # Register in database if app available
                                        if app is not None:
                                            try:
                                                with app.app_context():
                                                    alert_service.create_alert(alert_data)
                                                    print(f"[FORENSIC ANALYZER] Single Confirmation Alert {alert_uuid} registered in database for case {case_id} (Score: {sim*100:.1f}%).")
                                            except Exception as e:
                                                print(f"[FORENSIC ANALYZER] Error saving alert: {e}")

                                        # Register in Re-ID associator
                                        app_feat = reid_extractor_service.extract_descriptor(crop)
                                        cross_camera_service.register_tracklet(
                                            job.camera_id, track_id, app_feat, time.time(), face_match_case=match["case_id"]
                                        )

                                    elif sim > best_similarity_by_case.get(case_id, 0.0):
                                        # Child observed again with a higher confidence / clearer face crop!
                                        # Update the single existing alert without creating a duplicate alert.
                                        best_similarity_by_case[case_id] = sim
                                        prev_alert = alert_records_by_case.get(case_id)
                                        if prev_alert:
                                            prev_alert["similarity_score"] = sim
                                            prev_alert["track_id"] = track_id
                                            prev_alert["camera_location"] = f"{job.location} [Video Time: {time_str}]"
                                            
                                            # Overwrite snapshot files with the clearer / higher-scoring face crop
                                            cv2.imwrite(prev_alert["cctv_frame_path"], frame)
                                            cv2.imwrite(prev_alert["face_crop_path"], fcrop)

                                            if app is not None:
                                                try:
                                                    with app.app_context():
                                                        from app.database.models import PotentialMatchAlert, db
                                                        db_alert = PotentialMatchAlert.query.filter_by(alert_id=prev_alert["alert_id"]).first()
                                                        if db_alert and db_alert.status == "New":
                                                            db_alert.similarity_score = sim
                                                            db_alert.track_id = track_id
                                                            db_alert.camera_location = prev_alert["camera_location"]
                                                            db.session.commit()
                                                            print(f"[FORENSIC ANALYZER] Refined Alert {prev_alert['alert_id']} for case {case_id} with higher score {sim*100:.1f}%.")
                                                except Exception as e:
                                                    print(f"[FORENSIC ANALYZER] Error updating alert: {e}")
                                break

                    # Draw bounding box and label
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    cv2.rectangle(annotated, (x1, y1 - 24), (x1 + len(label) * 11, y1), box_color, -1)
                    cv2.putText(annotated, label, (x1 + 3, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

                    # Periodically record keyframes for officer frame review (every 30 frames or when match occurs)
                    if (frame_count % 30 == 0 or is_match) and crop is not None:
                        kf_name = f"kf_{job.job_id}_f{frame_count}_tr{track_id}.jpg"
                        kf_path = f"data/keyframes/{kf_name}"
                        cv2.imwrite(kf_path, crop)
                        job.keyframes.append({
                            "keyframe_id": kf_name,
                            "frame_number": frame_count,
                            "video_timestamp": time_str,
                            "track_id": track_id,
                            "thumbnail_url": f"/api/footage/keyframe/{kf_name}",
                            "quality": face_quality_desc,
                            "is_match": is_match,
                            "matched_case": matched_case["case_id"] if matched_case else None,
                            "similarity": round(matched_case["similarity"] * 100, 1) if matched_case else None
                        })

                # Write processed frame to output video
                if out_writer is not None and out_writer.isOpened():
                    out_writer.write(annotated)

                # Store latest frame JPEG for live preview polling in browser
                ret_enc, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret_enc:
                    job.latest_frame_jpeg = buf.tobytes()

                # Update performance metrics
                dur = time.time() - t0
                latencies.append(dur * 1000.0)
                if len(latencies) > 30:
                    latencies.pop(0)
                job.latency_ms = sum(latencies) / len(latencies)
                job.processing_fps = 1000.0 / job.latency_ms if job.latency_ms > 0 else 0.0
                job.progress_percent = min(100.0, (frame_count / job.total_frames) * 100.0)

            # Mark completed if not cancelled
            if job.status != "cancelled":
                job.status = "completed"
                job.progress_percent = 100.0
                job.output_video_path = output_path
                print(f"[FORENSIC ANALYZER] Job {job.job_id} COMPLETED. Output: {output_path}")

        except Exception as e:
            job.status = "error"
            job.error_message = str(e)
            print(f"[FORENSIC ANALYZER] Error during job {job.job_id}: {e}")
        finally:
            cap.release()
            if out_writer is not None:
                out_writer.release()
            job.end_time = time.time()

forensic_analyzer = ForensicVideoAnalyzer()
