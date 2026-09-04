import argparse
import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO

def evaluate_yolo(
    weights: str = "models/yolo/best.pt",
    data_yaml: str = "datasets/cctv_pedestrians/data.yaml",
    imgsz: int = 416,
    batch: int = 4,
    save_json: str = "runs/evaluation_report.json"
):
    print("=" * 65)
    print("  YOLO11n INDEPENDENT TEST SET EVALUATION - CPU")
    print("=" * 65)
    if not os.path.exists(weights):
        fallback = "models/yolo/yolo11n.pt" if os.path.exists("models/yolo/yolo11n.pt") else "yolo11n.pt"
        print(f"[NOTE] Checkpoint '{weights}' not found. Evaluating base checkpoint '{fallback}'.")
        weights = fallback

    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        split="test",
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

    eval_results = {
        "model_weights": weights,
        "evaluation_split": "test",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "f1_score": round(f1, 4),
        "save_dir": str(metrics.save_dir)
    }

    print("\n" + "=" * 65)
    print("  FINAL HELD-OUT TEST EVALUATION REPORT")
    print("=" * 65)
    print(f"  Precision:       {precision * 100:.2f}%")
    print(f"  Recall:          {recall * 100:.2f}%")
    print(f"  mAP@50:          {map50 * 100:.2f}%")
    print(f"  mAP@50-95:       {map50_95 * 100:.2f}%")
    print(f"  F1 Score:        {f1 * 100:.2f}%")
    print("-" * 65)
    # Honest performance analysis against 96% target (Section 16)
    target = 0.96
    print("  PERFORMANCE TARGET ANALYSIS (Target > 96.0%):")
    for metric_name, val in [("Precision", precision), ("Recall", recall), ("mAP@50", map50), ("mAP@50-95", map50_95)]:
        status = "PASSED" if val >= target else f"NEEDS IMPROVEMENT ({val*100:.1f}% vs 96.0%)"
        print(f"    - {metric_name:<10}: {status}")
    print("=" * 65)

    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"[REPORT] Saved test evaluation report to {save_json}")

    return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/yolo/best.pt")
    parser.add_argument("--data", type=str, default="datasets/cctv_pedestrians/data.yaml")
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()

    evaluate_yolo(args.weights, args.data, args.imgsz)
