import os
import cv2
import time
import uuid
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable

from app.config import Config
from app.ai.hardware import RealTimeProfiler, HardwareProfile
from app.ai.tracking.tracker import PersonTracker
from app.ai.face import face_detector_service, face_quality_service, face_recognizer_service
from app.ai.enhancement.enhancer import face_enhancer_service
from app.ai.reid.feature_extractor import reid_extractor_service
from app.ai.reid.association import cross_camera_service

class StreamWorker:
    """Asynchronous CPU video and RTSP stream processor.
    Runs detection, tracking, face matching, Re-ID, and alert generation in a background thread.
    """

    def __init__(self, source: str, camera_id: str = "CAM-01", camera_location: str = "Main Entrance"):
        self.source = source
        self.camera_id = camera_id
        self.camera_location = camera_location
        self.tracker = PersonTracker(model_path=Config.YOLO_MODEL_PATH)
        self.profiler = RealTimeProfiler()
        
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Source classification and health tracking
        is_net_stream = str(self.source).startswith(("rtsp://", "rtsps://", "http://", "https://"))
        self.source_type = "rtsp" if is_net_stream else ("file" if os.path.exists(str(self.source)) else "demo")
        self.stream_status = "idle"  # idle, connecting, online, fallback_demo, stopped, error
        self.is_fallback = False
        self.stream_error: Optional[str] = None
        self.consecutive_read_failures = 0
        
        self.current_frame_annotated: Optional[np.ndarray] = None
        self.latest_alert: Optional[dict] = None
        self.alerts_generated: List[dict] = []
        self.alerted_matches = set()
        self.total_frames = 0
        self.active_tracks_count = 0
        
        # Callback for saving alerts to database (injected by Flask service)
        self.on_alert_callback: Optional[Callable[[dict], None]] = None

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.stream_status = "connecting"
            self.thread = threading.Thread(target=self._process_loop, daemon=True)
            self.thread.start()
            print(f"[STREAM WORKER] Started processing stream for {self.camera_id} ({self.source})")

    def stop(self):
        self.is_running = False
        self.stream_status = "stopped"
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print(f"[STREAM WORKER] Stopped stream {self.camera_id}")

    def _generate_synthetic_cctv_frame(self, frame_idx: int) -> np.ndarray:
        """Generates a realistic CCTV stream frame with moving pedestrians if no video file exists."""
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Asphalt and background
        cv2.rectangle(frame, (0, int(h * 0.45)), (w, h), (80, 80, 80), -1)
        cv2.rectangle(frame, (0, 0), (w, int(h * 0.45)), (140, 125, 110), -1)

        # Walking pedestrian moving horizontally across CCTV view
        px = int(100 + (frame_idx * 4) % (w - 200))
        py = 320
        pw, ph = 90, 240

        # Person figure
        color_coat = (180, 70, 60)
        skin = (170, 200, 225)
        # Head
        cv2.circle(frame, (px + 45, py + 30), 22, skin, -1)
        # Eyes and mouth to simulate face
        cv2.circle(frame, (px + 38, py + 26), 3, (40, 40, 40), -1)
        cv2.circle(frame, (px + 52, py + 26), 3, (40, 40, 40), -1)
        cv2.line(frame, (px + 40, py + 38), (px + 50, py + 38), (30, 30, 150), 2)
        # Torso
        cv2.rectangle(frame, (px + 15, py + 52), (px + 75, py + 160), color_coat, -1)
        # Legs
        cv2.rectangle(frame, (px + 20, py + 160), (px + 40, py + ph), (50, 50, 60), -1)
        cv2.rectangle(frame, (px + 50, py + 160), (px + 70, py + ph), (50, 50, 60), -1)

        # Another pedestrian walking in opposite direction
        px2 = int((w - 200) - (frame_idx * 3) % (w - 300))
        py2 = 360
        cv2.circle(frame, (px2 + 40, py2 + 25), 20, skin, -1)
        cv2.rectangle(frame, (px2 + 15, py2 + 45), (px2 + 65, py2 + 150), (60, 140, 80), -1)
        cv2.rectangle(frame, (px2 + 20, py2 + 150), (px2 + 35, py2 + 210), (40, 40, 40), -1)
        cv2.rectangle(frame, (px2 + 45, py2 + 150), (px2 + 60, py2 + 210), (40, 40, 40), -1)

        # CCTV timestamp and camera OSD
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        status_tag = "[DEMO FALLBACK]" if self.is_fallback else "[LIVE]"
        tag_color = (0, 165, 255) if self.is_fallback else (0, 255, 255)
        cv2.putText(frame, f"{self.camera_id} [{self.camera_location}] - {ts} {status_tag}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tag_color, 2)
        return frame

    def _process_loop(self):
        cap = None
        use_synth = False
        source_str = str(self.source)

        if os.path.exists(source_str):
            try:
                cap = cv2.VideoCapture(source_str)
                if not cap.isOpened():
                    use_synth = True
                    self.is_fallback = True
                    self.stream_status = "fallback_demo"
                    self.stream_error = "Could not open video file; using synthetic fallback."
                else:
                    self.stream_status = "online"
                    self.is_fallback = False
            except Exception as e:
                use_synth = True
                self.is_fallback = True
                self.stream_status = "fallback_demo"
                self.stream_error = str(e)
        elif source_str.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            try:
                import socket
                from urllib.parse import urlparse
                parsed = urlparse(source_str)
                hostname = parsed.hostname
                port = parsed.port or (554 if parsed.scheme.lower() in ("rtsp", "rtsps") else 80)
                
                host_reachable = False
                if hostname:
                    try:
                        sock = socket.create_connection((hostname, port), timeout=1.5)
                        sock.close()
                        host_reachable = True
                    except Exception as e:
                        print(f"[STREAM WORKER] {self.camera_id}: Host {hostname}:{port} unreachable ({e}). Immediate fallback to demo.")
                        host_reachable = False

                if not host_reachable:
                    use_synth = True
                    self.is_fallback = True
                    self.stream_status = "fallback_demo"
                    self.stream_error = f"RTSP host {hostname}:{port} unreachable. Demo fallback active."
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(source_str)

                    if cap is None or not cap.isOpened():
                        use_synth = True
                        self.is_fallback = True
                        self.stream_status = "fallback_demo"
                        self.stream_error = "RTSP camera stream handshake rejected. Active synthetic fallback."
                        print(f"[STREAM WORKER] {self.camera_id}: RTSP handshake failed, using synthetic fallback.")
                    else:
                        self.stream_status = "online"
                        self.is_fallback = False
                        self.stream_error = None
                        print(f"[STREAM WORKER] {self.camera_id}: Connected to RTSP stream.")
            except Exception as e:
                use_synth = True
                self.is_fallback = True
                self.stream_status = "fallback_demo"
                self.stream_error = str(e)
        else:
            use_synth = True
            self.stream_status = "online"
            self.is_fallback = False

        frame_idx = 0
        while self.is_running:
            start_t = time.time()
            frame = None

            if not use_synth and cap is not None and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # If video file reached end, loop it
                        if self.source_type == "file":
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        
                        if not ret or frame is None:
                            self.consecutive_read_failures += 1
                            if self.consecutive_read_failures >= 5:
                                use_synth = True
                                self.is_fallback = True
                                self.stream_status = "fallback_demo"
                                self.stream_error = "Stream frame timeout; falling back to demo simulation."
                                frame = self._generate_synthetic_cctv_frame(frame_idx)
                            else:
                                time.sleep(0.04)
                                continue
                        else:
                            self.consecutive_read_failures = 0
                            self.stream_status = "online"
                            self.is_fallback = False
                    else:
                        self.consecutive_read_failures = 0
                        self.stream_status = "online"
                        self.is_fallback = False
                except Exception as e:
                    use_synth = True
                    self.is_fallback = True
                    self.stream_status = "fallback_demo"
                    self.stream_error = str(e)
                    frame = self._generate_synthetic_cctv_frame(frame_idx)
            else:
                frame = self._generate_synthetic_cctv_frame(frame_idx)
                time.sleep(0.03)  # Emulate ~30fps source

            frame_idx += 1
            self.total_frames += 1

            if frame is not None and self.current_frame_annotated is None:
                with self.lock:
                    self.current_frame_annotated = frame.copy()

            # 1. Update Tracking on CPU
            tracked = self.tracker.update(frame)
            self.active_tracks_count = len(tracked)
            annotated = frame.copy()

            # 2. Analyze Tracked Persons
            for obj in tracked:
                track_id = obj["track_id"]
                x1, y1, x2, y2 = obj["bbox"]
                crop = obj["crop"]
                should_check_face = obj["should_check_face"]
                trajectory = obj["trajectory"]

                # Draw trajectory trail
                for pt in trajectory[-15:]:
                    cv2.circle(annotated, pt, 2, (0, 255, 255), -1)

                box_color = (0, 255, 0) # Normal tracking green
                label = f"ID: {track_id}"

                # Periodic Face Recognition check
                if should_check_face and crop is not None:
                    # Face detection within person crop
                    face_dets = face_detector_service.detect_faces(crop)
                    for fdet in face_dets:
                        fcrop = fdet["crop"]
                        q_res = face_quality_service.assess_crop(fcrop)
                        
                        # Selective Face Enhancement
                        if face_enhancer_service.should_enhance(fcrop, q_res["score"]):
                            fcrop = face_enhancer_service.enhance_face(fcrop)
                            q_res = face_quality_service.assess_crop(fcrop)

                        if q_res["sufficient"]:
                            # Generate AdaFace 512-D embedding
                            emb = face_recognizer_service.generate_embedding(fcrop)
                            # Compare against active missing child cases
                            match = face_recognizer_service.determine_candidate(emb)
                            
                            if match:
                                box_color = (0, 0, 255) # Red alert
                                label = f"MATCH: {match['case_id']} ({match['similarity']*100:.1f}%)"

                                # One alert per case/track appearance; repeated face checks
                                # on the same tracked person must not create alert spam.
                                alert_key = (match["case_id"], track_id)
                                if alert_key in self.alerted_matches:
                                    break
                                self.alerted_matches.add(alert_key)
                                
                                # Snapshot paths
                                alert_uuid = f"ALT-{int(time.time())}-{track_id}"
                                frame_path = f"data/snapshots/frame_{alert_uuid}.jpg"
                                face_path = f"data/snapshots/face_{alert_uuid}.jpg"
                                os.makedirs("data/snapshots", exist_ok=True)
                                cv2.imwrite(frame_path, frame)
                                cv2.imwrite(face_path, fcrop)

                                alert_data = {
                                    "alert_id": alert_uuid,
                                    "case_id": match["case_id"],
                                    "camera_id": self.camera_id,
                                    "camera_location": self.camera_location,
                                    "track_id": track_id,
                                    "similarity_score": match["similarity"],
                                    "cctv_frame_path": frame_path,
                                    "face_crop_path": face_path,
                                    "reference_photo_path": match["metadata"].get("photo_path", ""),
                                    "model_version": Config.FACE_MODEL_VERSION,
                                    "status": "New",
                                    "timestamp": time.time()
                                }
                                self.latest_alert = alert_data
                                self.alerts_generated.append(alert_data)
                                if self.on_alert_callback:
                                    try:
                                        self.on_alert_callback(alert_data)
                                    except Exception as e:
                                        print(f"[ERROR] Alert callback error: {e}")

                                # Register in Re-ID associator
                                app_feat = reid_extractor_service.extract_descriptor(crop)
                                cross_camera_service.register_tracklet(
                                    self.camera_id, track_id, app_feat, time.time(), face_match_case=match["case_id"]
                                )
                                break

                # Draw bounding box and label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                cv2.rectangle(annotated, (x1, y1 - 25), (x1 + len(label) * 11, y1), box_color, -1)
                cv2.putText(annotated, label, (x1 + 4, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            duration = time.time() - start_t
            self.profiler.tick(duration)

            # Draw real-time CPU telemetry overlay
            telemetry = self.profiler.get_summary()
            osd_text = f"FPS: {telemetry['fps']} | Latency: {telemetry['latency_ms']}ms | CPU: {telemetry['cpu_percent']}% | Tracks: {self.active_tracks_count}"
            cv2.putText(annotated, osd_text, (30, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            with self.lock:
                self.current_frame_annotated = annotated

        if cap is not None:
            cap.release()

    def get_latest_frame_bytes(self) -> Optional[bytes]:
        with self.lock:
            if self.current_frame_annotated is None:
                return None
            ret, buf = cv2.imencode(".jpg", self.current_frame_annotated)
            return buf.tobytes() if ret else None

    def get_status(self) -> dict:
        status = self.profiler.get_summary()
        status.update({
            "camera_id": self.camera_id,
            "camera_location": self.camera_location,
            "source": self.source,
            "source_type": self.source_type,
            "stream_status": self.stream_status,
            "is_fallback": self.is_fallback,
            "stream_error": self.stream_error,
            "is_running": self.is_running,
            "active_tracks": self.active_tracks_count,
            "alerts_count": len(self.alerts_generated),
            "latest_alert": self.latest_alert
        })
        return status
