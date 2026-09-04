import argparse
import time
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.hardware import HardwareProfile
from ultralytics import YOLO

def benchmark_cpu_training(data_yaml: str, imgsz: int = 416, batch: int = 4, target_epochs: int = 10):
    HardwareProfile.print_cpu_training_banner("YOLO11n CPU Pre-Training Benchmark")
    
    print(f"\n[BENCHMARK] Running 1-epoch calibration with imgsz={imgsz}, batch={batch} on CPU...")
    model = YOLO("models/yolo/yolo11n.pt" if os.path.exists("models/yolo/yolo11n.pt") else "yolo11n.pt")
    
    start = time.time()
    try:
        # Run 1 epoch to measure throughput
        results = model.train(
            data=data_yaml,
            epochs=1,
            imgsz=imgsz,
            batch=batch,
            device="cpu",
            workers=2,
            project="runs/benchmarks",
            name="cpu_calibration",
            verbose=False,
            plots=False
        )
        duration_per_epoch = time.time() - start
        estimated_total_sec = duration_per_epoch * target_epochs
        estimated_hours = round(estimated_total_sec / 3600.0, 2)
        estimated_minutes = round(estimated_total_sec / 60.0, 1)

        print("\n" + "=" * 65)
        print("  CPU BENCHMARK RESULTS")
        print("=" * 65)
        print(f"  Calibrated 1-Epoch Duration: {round(duration_per_epoch, 2)} seconds")
        print(f"  Target Epochs:              {target_epochs}")
        print(f"  Estimated Total Training:   {estimated_minutes} minutes (~{estimated_hours} hours)")
        print("=" * 65)
        return duration_per_epoch
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11n CPU Pre-Training Benchmark")
    parser.add_argument("--data", type=str, default="datasets/cctv_pedestrians/data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Target epochs")
    args = parser.parse_args()

    benchmark_cpu_training(args.data, args.imgsz, args.batch, args.epochs)
