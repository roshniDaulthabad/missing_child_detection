import os
import sys
import time
import cv2
import numpy as np
import pytest
from pathlib import Path
from io import BytesIO
from werkzeug.datastructures import FileStorage

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app
from app.config import Config
from app.database.models import db, MissingChild, PotentialMatchAlert
from app.services.case_service import case_service
from app.ai.pipeline.video_forensics import forensic_analyzer

def generate_synthetic_mp4(dest_path: str, num_frames: int = 40):
    """Generates a small test video with a person walking across the frame."""
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dest_path, fourcc, 20.0, (w, h))

    for f in range(num_frames):
        frame = np.full((h, w, 3), 70, dtype=np.uint8)
        # Asphalt and wall
        cv2.rectangle(frame, (0, int(h * 0.4)), (w, h), (90, 90, 90), -1)

        # Moving person
        px = int(50 + f * 10)
        py = 180
        pw, ph = 70, 220

        # Person figure
        skin = (170, 200, 225)
        # Head
        cv2.circle(frame, (px + 35, py + 25), 18, skin, -1)
        cv2.circle(frame, (px + 29, py + 22), 2, (40, 40, 40), -1)
        cv2.circle(frame, (px + 41, py + 22), 2, (40, 40, 40), -1)
        cv2.line(frame, (px + 31, py + 33), (px + 39, py + 33), (30, 30, 160), 2)
        # Torso
        cv2.rectangle(frame, (px + 10, py + 45), (px + 60, py + 140), (200, 80, 60), -1)
        # Legs
        cv2.rectangle(frame, (px + 15, py + 140), (px + 32, py + ph), (40, 40, 50), -1)
        cv2.rectangle(frame, (px + 38, py + 140), (px + 55, py + ph), (40, 40, 50), -1)

        out.write(frame)

    out.release()
    print(f"[TEST SETUP] Generated synthetic video at {dest_path} ({num_frames} frames)")

def test_true_video_forensic_pipeline():
    with app.app_context():
        # 1. Generate test video
        os.makedirs("data/uploads/footage", exist_ok=True)
        test_video_path = "data/uploads/footage/forensic_test.mp4"
        generate_synthetic_mp4(test_video_path, num_frames=30)
        assert os.path.exists(test_video_path)

        # 2. Register child matching the synthetic face
        face_img = np.full((120, 100, 3), 180, dtype=np.uint8)
        cv2.circle(face_img, (35, 45), 8, (50, 50, 50), -1)
        cv2.circle(face_img, (65, 45), 8, (50, 50, 50), -1)
        cv2.line(face_img, (35, 85), (65, 85), (40, 40, 150), 3)
        _, buf = cv2.imencode(".jpg", face_img)
        storage = FileStorage(stream=BytesIO(buf.tobytes()), filename="forensic_child.jpg", content_type="image/jpeg")

        case_id = f"MC-FORENSIC-{int(time.time())}"
        child = case_service.register_child({
            "case_id": case_id,
            "child_name": "Forensic Verification Child",
            "age": 9,
            "gender": "Male",
            "parent_name": "Test Parent",
            "parent_contact": "555-0199",
            "date_missing": "2026-09-04",
            "last_known_location": "Platform 1"
        }, storage)

        # 3. Launch Forensic Analysis Job
        app_ctx = app.app_context()
        job_id = forensic_analyzer.start_job(
            test_video_path,
            camera_id="CAM-FORENSIC-01",
            location="Forensic Platform 1",
            app_context=app_ctx
        )
        assert job_id is not None

        # 4. Wait for job to process
        max_wait = 30
        t0 = time.time()
        job = None
        while time.time() - t0 < max_wait:
            job = forensic_analyzer.get_job(job_id)
            if job and job.status in ["completed", "error"]:
                break
            time.sleep(1.0)

        assert job is not None
        print(f"[TEST RESULT] Job {job_id} Status: {job.status}, Processed: {job.current_frame}/{job.total_frames}")
        assert job.status == "completed"
        assert job.current_frame == job.total_frames
        assert job.progress_percent == 100.0
        assert job.output_video_path is not None
        assert os.path.exists(job.output_video_path)
        assert job.error_message is None
