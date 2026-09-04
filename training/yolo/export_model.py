import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO

def export_yolo(weights: str = "models/yolo/best.pt", format: str = "torchscript", imgsz: int = 416):
    print("=" * 65)
    print(f"  YOLO11n MODEL EXPORT ({format.upper()})")
    print("=" * 65)
    if not os.path.exists(weights):
        fallback = "models/yolo/yolo11n.pt" if os.path.exists("models/yolo/yolo11n.pt") else "yolo11n.pt"
        print(f"[NOTE] Checkpoint '{weights}' not found. Exporting base checkpoint '{fallback}'.")
        weights = fallback

    model = YOLO(weights)
    export_path = model.export(
        format=format,
        imgsz=imgsz
    )
    print(f"\n[SUCCESS] Model exported to {format.upper()}: {export_path}")
    return export_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/yolo/best.pt")
    parser.add_argument("--format", type=str, default="torchscript")
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()

    export_yolo(args.weights, args.format, args.imgsz)
