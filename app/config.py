import os
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load optional .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    BASE_DIR = BASE_DIR
    # Flask configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "hackwave-police-secret-key-2026-production-ready")
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Database: SQLite fallback, PostgreSQL production ready
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'data' / 'hackwave.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Hardware constraint: STRICT CPU ONLY
    DEVICE = "cpu"
    NUM_CPU_WORKERS = int(os.getenv("NUM_CPU_WORKERS", "2"))

    # Processing Modes: PERFORMANCE, BALANCED, ACCURACY
    OPERATION_MODE = os.getenv("OPERATION_MODE", "BALANCED").upper()

    MODE_SETTINGS = {
        "PERFORMANCE": {
            "imgsz": 416,
            "face_recognition_interval": 15,
            "enable_enhancement": False,
            "conf_threshold": 0.40,
            "tracker": "bytetrack.yaml"
        },
        "BALANCED": {
            "imgsz": 512,
            "face_recognition_interval": 10,
            "enable_enhancement": True,
            "conf_threshold": 0.45,
            "tracker": "botsort.yaml"
        },
        "ACCURACY": {
            "imgsz": 640,
            "face_recognition_interval": 5,
            "enable_enhancement": True,
            "conf_threshold": 0.50,
            "tracker": "botsort.yaml"
        }
    }

    # Model paths
    MODELS_DIR = BASE_DIR / "models"
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(MODELS_DIR / "yolo" / "yolo11n.pt"))
    FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH", str(MODELS_DIR / "face" / "adaface_iresnet50.pt"))
    REID_MODEL_PATH = os.getenv("REID_MODEL_PATH", str(MODELS_DIR / "reid" / "osnet_reid.pt"))

    # Model Versioning for Audit Trail
    YOLO_VERSION = "yolo11n-cctv-cpu-v1"
    FACE_MODEL_VERSION = "adaface-iresnet-cpu-v1"
    REID_MODEL_VERSION = "cpu-reid-appearance-v1"

    # AI Detection & Matching Thresholds
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.62"))
    REID_SIMILARITY_THRESHOLD = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.58"))
    FACE_MIN_SIZE = int(os.getenv("FACE_MIN_SIZE", "24"))  # pixels (adapted for surveillance crops)
    FACE_SHARPNESS_MIN = float(os.getenv("FACE_SHARPNESS_MIN", "18.0"))  # Laplacian variance (adapted for CCTV)

    # Security & Storage
    DATA_DIR = BASE_DIR / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"
    ENCRYPTED_DIR = DATA_DIR / "encrypted"
    SNAPSHOTS_DIR = DATA_DIR / "snapshots"

    # AES-256-GCM Key (32 bytes) - Generated once if not supplied in env
    _default_key = base64.b64encode(b"HackwaveSecure256BitKeyForChildWelfare2026!").decode()[:44]
    AES_SECRET_KEY = os.getenv("AES_SECRET_KEY", _default_key)

    # Demo Mode
    DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() in ("true", "1", "yes")

    # Featherless.ai LLM Integration
    FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY", "")
    FEATHERLESS_BASE_URL = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
    FEATHERLESS_MODEL = os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen3.8-Flash-Next")

    @classmethod
    def get_active_mode_settings(cls):
        mode = cls.OPERATION_MODE if cls.OPERATION_MODE in cls.MODE_SETTINGS else "BALANCED"
        return cls.MODE_SETTINGS[mode]
