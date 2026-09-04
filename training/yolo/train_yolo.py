import argparse
import json
import time
import os
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.ai.hardware import HardwareProfile
from ultralytics import YOLO

def train_yolo(
    data_yaml: str = "datasets/cctv_pedestrians/data.yaml",
    weights: str = "models/yolo/yolo11n.pt",
    epochs: int = 5,
    imgsz: int = 416,
    batch: int = 4,
    lr: float = 0.001,
    optimizer: str = "auto",
    workers: int = 2,
    patience: int = 5,
    device: str = "cpu",
    project: str = "runs/train",
    name: str = "yolo11n_cctv_cpu"
):
    # Enforce CPU execution
    if device.lower() != "cpu":
        print(f"[WARNING] Non-CPU device requested ({device}), but this system is constrained to STRICT CPU execution. Overriding to 'cpu'.")
        device = "cpu"

    # Display computational profile banner
    HardwareProfile.print_cpu_training_banner("YOLO11n CCTV Domain-Adapted Fine-Tuning")
    
    start_time = time.time()
    
    # Resolve weights path
    if not os.path.exists(weights):
        if os.path.exists("yolo11n.pt"):
            weights = "yolo11n.pt"
        else:
            weights = "yolo11n.pt"  # Will be auto-downloaded by ultralytics

    print(f"\n[CONFIG] Checkpoint:    {weights}")
    print(f"[CONFIG] Dataset YAML:  {data_yaml}")
    print(f"[CONFIG] Epochs:        {epochs}")
    print(f"[CONFIG] Image Size:    {imgsz}")
    print(f"[CONFIG] Batch Size:    {batch}")
    print(f"[CONFIG] Learning Rate: {lr}")
    print(f"[CONFIG] Workers:       {workers}")
    print(f"[CONFIG] Optimizer:     {optimizer}")
    print(f"[CONFIG] Device:        {device}\n")

    # Load model
    model = YOLO(weights)

    # Train model on CPU
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        optimizer=optimizer,
        workers=workers,
        patience=patience,
        device="cpu",
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
        save=True,
        # CCTV Domain-Specific Augmentation
        hsv_h=0.015,     # subtle color variation
        hsv_s=0.5,       # saturation changes
        hsv_v=0.4,       # brightness changes (day/night)
        degrees=5.0,     # slight camera tilt
        translate=0.1,   # camera frame shift
        scale=0.3,       # scale / distance variation
        shear=2.0,       # perspective shear
        fliplr=0.5,      # horizontal walking direction
        mosaic=0.5,      # multi-person context
        mixup=0.0        # keep CCTV realism
    )

    training_duration = time.time() - start_time
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / name

    # Copy best weights to models/yolo/best.pt for easy application use
    best_weights = save_dir / "weights" / "best.pt"
    target_best = BASE_DIR / "models" / "yolo" / "best.pt"
    if best_weights.exists():
        import shutil
        target_best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, target_best)
        print(f"\n[MODEL EXPORT] Copied best model weights to: {target_best}")

    # Extract final metrics if available
    metrics_summary = {}
    if hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            metrics_summary[k] = round(float(v), 4)

    # Experiment tracking JSON
    exp_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "YOLO11n-CCTV",
        "device": "cpu",
        "training_duration_seconds": round(training_duration, 2),
        "training_duration_minutes": round(training_duration / 60.0, 2),
        "hyperparameters": {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "learning_rate": lr,
            "optimizer": optimizer,
            "workers": workers
        },
        "save_directory": str(save_dir),
        "best_weights": str(best_weights) if best_weights.exists() else None,
        "metrics": metrics_summary
    }

    record_file = save_dir / "experiment_record.json"
    with open(record_file, "w") as f:
        json.dump(exp_record, f, indent=2)

    print("=" * 65)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"  Duration:         {exp_record['training_duration_minutes']} minutes")
    print(f"  Save Directory:   {save_dir}")
    print(f"  Best Checkpoint:  {best_weights}")
    print(f"  Experiment Log:   {record_file}")
    print("=" * 65)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11n CCTV Fine-Tuning (CPU Only)")
    parser.add_argument("--data", type=str, default="datasets/cctv_pedestrians/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="models/yolo/yolo11n.pt", help="Pretrained weights")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (conservative CPU default)")
    parser.add_argument("--imgsz", type=int, default=416, help="Image resolution for training")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cpu", help="Device (forced to CPU)")

    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr=args.lr,
        optimizer=args.optimizer,
        workers=args.workers,
        patience=args.patience,
        device=args.device
    )
