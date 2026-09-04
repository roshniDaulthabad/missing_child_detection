import pytest
import time
from app import create_app
from app.database.models import db, Camera
from app.ai.pipeline import worker_pool
from app.ai.pipeline.processor import StreamWorker

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.app_context():
        # Ensure test camera does not already exist
        Camera.query.filter_by(camera_id="CAM-TEST-01").delete()
        db.session.commit()

        # Seed test camera
        cam = Camera(
            camera_id="CAM-TEST-01",
            name="Test Gate CCTV",
            location="Gate 1 Test",
            latitude=28.6,
            longitude=77.2,
            fps=15.0,
            status="online"
        )
        db.session.add(cam)
        db.session.commit()

        yield app.test_client()

        worker_pool.stop_all()
        Camera.query.filter_by(camera_id="CAM-TEST-01").delete()
        db.session.commit()

def test_validate_rtsp_empty(client):
    res = client.post("/api/stream/validate_rtsp", json={})
    assert res.status_code == 400
    data = res.get_json()
    assert data["valid"] is False

def test_validate_rtsp_invalid_scheme(client):
    res = client.post("/api/stream/validate_rtsp", json={"rtsp_url": "ftp://192.168.1.10/video"})
    assert res.status_code == 400
    data = res.get_json()
    assert data["valid"] is False
    assert "Unsupported scheme" in data["message"]

def test_validate_rtsp_unreachable_network(client):
    # RFC 5737 TEST-NET address 192.0.2.1 is unroutable; socket probe will timeout quickly
    res = client.post("/api/stream/validate_rtsp", json={"rtsp_url": "rtsp://192.0.2.1:554/stream1"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["valid"] is True
    assert data["reachable"] is False
    assert "timed out" in data["message"].lower() or "network" in data["message"].lower()

def test_camera_rtsp_save_and_masking(client):
    rtsp_url = "rtsp://admin:SecretPass123@192.168.1.50:554/h264"
    res = client.post("/api/cameras/CAM-TEST-01/rtsp", json={"rtsp_url": rtsp_url})
    assert res.status_code == 200

    # Query cameras list to verify masking
    cams_res = client.get("/api/cameras")
    assert cams_res.status_code == 200
    cams = cams_res.get_json()
    test_cam = next((c for c in cams if c["camera_id"] == "CAM-TEST-01"), None)
    assert test_cam is not None
    assert test_cam["has_rtsp"] is True
    # Password must be masked in masked_rtsp
    assert "SecretPass123" not in test_cam["masked_rtsp"]
    assert ":****@" in test_cam["masked_rtsp"]

def test_stream_start_stop_demo(client):
    start_res = client.post("/api/stream/start", json={
        "camera_id": "CAM-TEST-01",
        "source": "demo"
    })
    assert start_res.status_code == 200
    assert start_res.get_json()["status"] == "running"

    status_res = client.get("/api/stream/status?camera_id=CAM-TEST-01")
    assert status_res.status_code == 200
    stat = status_res.get_json()
    assert stat["is_running"] is True
    assert stat["source_type"] == "demo"

    stop_res = client.post("/api/stream/stop", json={"camera_id": "CAM-TEST-01"})
    assert stop_res.status_code == 200
    assert stop_res.get_json()["status"] == "stopped"

def test_stream_worker_fallback_on_unreachable_rtsp():
    worker = StreamWorker(source="rtsp://192.0.2.1:554/nonexistent", camera_id="CAM-TEST-FALLBACK")
    assert worker.source_type == "rtsp"
    worker.start()
    time.sleep(2.0)
    stat = worker.get_status()
    worker.stop()
    # Must have fallen back to synthetic CCTV demo without crashing
    assert stat["is_fallback"] is True
    assert stat["stream_status"] == "fallback_demo"
    frame_bytes = worker.get_latest_frame_bytes()
    assert frame_bytes is not None

def test_stream_accepts_and_propagates_user_rtsp_url(client):
    test_rtsp = "rtsp://admin:pass@192.168.1.200:554/live"
    # 1. Start stream specifying custom RTSP URL
    res = client.post("/api/stream/start", json={
        "camera_id": "CAM-TEST-01",
        "source": test_rtsp
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "running"
    assert data["source"] == test_rtsp
    assert data["source_type"] == "rtsp"

    # 2. Check stream status returns RTSP source
    stat_res = client.get("/api/stream/status?camera_id=CAM-TEST-01")
    assert stat_res.status_code == 200
    stat = stat_res.get_json()
    assert stat["source"] == test_rtsp
    assert stat["source_type"] == "rtsp"

    # 3. Verify camera persisted the RTSP configuration
    cams_res = client.get("/api/cameras")
    cams = cams_res.get_json()
    cam = next((c for c in cams if c["camera_id"] == "CAM-TEST-01"), None)
    assert cam is not None
    assert cam["has_rtsp"] is True
    assert cam["rtsp_url"] == test_rtsp

    # 4. Switch stream back to demo
    switch_res = client.post("/api/stream/start", json={
        "camera_id": "CAM-TEST-01",
        "source": "demo"
    })
    assert switch_res.status_code == 200
    switch_stat = client.get("/api/stream/status?camera_id=CAM-TEST-01").get_json()
    assert switch_stat["source"] == "demo"
    assert switch_stat["source_type"] == "demo"

    # Clean up
    client.post("/api/stream/stop", json={"camera_id": "CAM-TEST-01"})

