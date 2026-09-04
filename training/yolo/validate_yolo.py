import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO

def validate_yolo(
    weights: str = "models/yolo/best.pt",
    data_yaml: str = "datasets/cctv_pedestrians/data.yaml",
    imgsz: int = 416,
    batch: int = 4
):
    print("=" * 65)
    print("  YOLO11n VALIDATION (VAL SPLIT) - CPU")
    print("=" * 65)
    if not os.path.exists(weights):
        fallback = "models/yolo/yolo11n.pt" if os.path.exists("models/yolo/yolo11n.pt") else "yolo11n.pt"
        print(f"[NOTE] Checkpoint '{weights}' not found. Falling back to '{fallback}'.")
        weights = fallback

    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        split="val",
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        plots=True,
        verbose=True
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n" + "-" * 65)
    print("  VALIDATION METRICS SUMMARY")
    print("-" * 65)
    print(f"  Precision:       {precision * 100:.2f}%")
    print(f"  Recall:          {recall * 100:.2f}%")
    print(f"  mAP@50:          {map50 * 100:.2f}%")
    print(f"  mAP@50-95:       {map50_95 * 100:.2f}%")
    print(f"  F1 Score:        {f1 * 100:.2f}%")
    print("-" * 65)
    print(f"  Plots & Curves:  {metrics.save_dir}")
    print("=" * 65)

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "f1": f1
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/yolo/best.pt")
    parser.add_argument("--data", type=str, default="datasets/cctv_pedestrians/data.yaml")
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()

    validate_yolo(args.weights, args.data, args.imgsz)
