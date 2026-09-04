import os
import sys
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app
from app.config import Config
from app.ai.hardware import HardwareProfile

if __name__ == "__main__":
    HardwareProfile.print_cpu_training_banner("EasyFind Missing Child Detection & Welfare System")
    print("\n[STARTUP] Initializing CPU-First Police Welfare Server...")
    print(f"[STARTUP] Operation Mode:   {Config.OPERATION_MODE}")
    print(f"[STARTUP] YOLO Model:       {Config.YOLO_MODEL_PATH}")
    print(f"[STARTUP] Face Model:       {Config.FACE_MODEL_PATH}")
    print(f"[STARTUP] Database:         {Config.SQLALCHEMY_DATABASE_URI}")
    print(f"[STARTUP] Demo Mode:        {Config.DEMO_MODE}")
    port = int(os.getenv("PORT", 5000))
    print(f"[STARTUP] Dashboard URL:    http://127.0.0.1:{port}\n")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
