import pytest
from app import create_app
from app.config import Config
from app.database.models import db, MissingChild, PotentialMatchAlert, AuditLog
from app.services.featherless_service import FeatherlessService, featherless_service
from app.services.alert_service import alert_service

@pytest.fixture
def test_app():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.app_context():
        db.create_all()
        # Create a sample missing child
        child = MissingChild(
            case_id="TEST-CASE-2026",
            child_name="Rohan Mehra",
            age=9,
            gender="Male",
            photo_path="data/uploads/test_rohan.jpg",
            photo_hash="sha3_mock_hash_123456",
            parent_name="Kavita Mehra",
            parent_contact="+91-9876543210",
            date_missing="2026-03-01",
            last_known_location="Sector 4 Central Park",
            physical_description="Red jacket, navy blue jeans",
            identifying_characteristics="Scar on right eyebrow",
            priority="Critical",
            status="Active"
        )
        db.session.add(child)
        db.session.commit()
        yield app
        db.session.rollback()
        PotentialMatchAlert.query.filter(PotentialMatchAlert.alert_id.like("ALT-%")).delete()
        MissingChild.query.filter_by(case_id="TEST-CASE-2026").delete()
        db.session.commit()

@pytest.fixture
def client(test_app):
    return test_app.test_client()

def test_featherless_service_configuration():
    assert featherless_service.base_url == "https://api.featherless.ai/v1"
    assert featherless_service.model == "mistralai/Mistral-7B-Instruct-v0.3"
    assert featherless_service.timeout == 8

def test_featherless_offline_fallback():
    # An unconfigured service instance must return a safe fallback without crashing
    unconfigured_service = FeatherlessService(api_key="")
    assert unconfigured_service.is_configured() is False

    child_dict = {"case_id": "TEST-01", "child_name": "Test Child"}
    alert_dict = {"camera_id": "CAM-01", "similarity_score": 0.82}
    
    result = unconfigured_service.assess_potential_match(child_dict, alert_dict)
    assert "status" in result
    assert "Offline Fallback" in result["status"]
    assert "confidence" in result
    assert "reasoning" in result
    assert "recommendation" in result

def test_featherless_live_assessment():
    if not featherless_service.is_configured():
        pytest.skip("Featherless API key not provided")

    child_dict = {
        "case_id": "TEST-CASE-2026",
        "child_name": "Rohan Mehra",
        "age": 9,
        "gender": "Male",
        "date_missing": "2026-03-01",
        "last_known_location": "Sector 4 Central Park",
        "physical_description": "Red jacket, navy blue jeans",
        "identifying_characteristics": "Scar on right eyebrow",
        "priority": "Critical"
    }
    alert_dict = {
        "alert_id": "ALT-TEST-LIVE",
        "case_id": "TEST-CASE-2026",
        "camera_id": "CAM-01",
        "camera_location": "Sector 4 Main Gate CCTV",
        "track_id": 5,
        "similarity_score": 0.84,
        "timestamp": "2026-03-02 10:15:00"
    }

    result = featherless_service.assess_potential_match(child_dict, alert_dict)
    assert "status" in result
    assert "confidence" in result
    assert "reasoning" in result
    assert "recommendation" in result
    assert len(result["reasoning"]) > 5
    assert len(result["recommendation"]) > 5

def test_alert_creation_with_featherless_integration(test_app):
    with test_app.app_context():
        alert_data = {
            "alert_id": "ALT-FEATHERLESS-001",
            "case_id": "TEST-CASE-2026",
            "camera_id": "CAM-01",
            "camera_location": "Sector 4 Entry",
            "track_id": 10,
            "similarity_score": 0.87,
            "cctv_frame_path": "data/snapshots/test_frame.jpg",
            "face_crop_path": "data/snapshots/test_crop.jpg",
            "reference_photo_path": "data/uploads/test_ref.jpg",
            "model_version": "adaface-iresnet-cpu-v1"
        }
        
        alert = alert_service.create_alert(alert_data)
        assert alert.alert_id == "ALT-FEATHERLESS-001"
        assert alert.ai_status is not None
        assert alert.ai_confidence is not None
        assert alert.ai_assessment is not None
        assert alert.ai_recommendation is not None
        # Automated status check
        assert any(term in alert.status for term in ["Featherless AI", "AI Fallback"])

        # Verify persisted in database
        queried = PotentialMatchAlert.query.filter_by(alert_id="ALT-FEATHERLESS-001").first()
        assert queried is not None
        assert queried.ai_status == alert.ai_status
        assert queried.status == alert.status

def test_reverify_endpoint(test_app, client):
    with test_app.app_context():
        alert_data = {
            "alert_id": "ALT-REVERIFY-001",
            "case_id": "TEST-CASE-2026",
            "camera_id": "CAM-02",
            "camera_location": "Platform 1 Exit",
            "track_id": 14,
            "similarity_score": 0.78,
            "cctv_frame_path": "data/snapshots/test_f2.jpg",
            "face_crop_path": "data/snapshots/test_c2.jpg",
            "reference_photo_path": "data/uploads/test_ref2.jpg",
            "model_version": "adaface-iresnet-cpu-v1"
        }
        alert_service.create_alert(alert_data)

    res = client.post("/api/alerts/ALT-REVERIFY-001/ai-verify")
    assert res.status_code == 200
    data = res.get_json()
    assert data["alert_id"] == "ALT-REVERIFY-001"
    assert "ai_status" in data
    assert "ai_confidence" in data
    assert "ai_assessment" in data
    assert "ai_recommendation" in data
