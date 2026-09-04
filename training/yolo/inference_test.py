import argparse
import time
import os
import sys
import cv2
import psutil
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO
from app.ai.hardware import HardwareProfile

def benchmark_inference(weights: str = "models/yolo/yolo11n.pt", num_frames: int = 30, imgsz: int = 416):
    HardwareProfile.print_cpu_training_banner("YOLO11n CPU Inference Benchmark")
    
    if not os.path.exists(weights):
        if os.path.exists("yolo11n.pt"):
            weights = "yolo11n.pt"

    print(f"\n[BENCHMARK] Loading PyTorch model '{weights}' on CPU...")
    model = YOLO(weights)
    
    # Generate representative CCTV test frame (720x1280)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "CAM-01 CCTV STREAM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # Draw simulated pedestrians
    cv2.rectangle(dummy_frame, (200, 250), (280, 550), (120, 150, 200), -1)
    cv2.circle(dummy_frame, (240, 220), 30, (180, 190, 210), -1)

    # Warmup
    print("[BENCHMARK] Warming up CPU inference engine (3 runs)...")
    for _ in range(3):
        model.predict(dummy_frame, imgsz=imgsz, device="cpu", verbose=False)

    print(f"[BENCHMARK] Profiling PyTorch CPU inference over {num_frames} frames (imgsz={imgsz})...")
    latencies = []
    for _ in range(num_frames):
        t0 = time.time()
        model.predict(dummy_frame, imgsz=imgsz, device="cpu", verbose=False)
        latencies.append((time.time() - t0) * 1000.0)

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    p95_latency = np.percentile(latencies, 95)

    ts_path = Path("models/yolo/best.torchscript")
    ts_results = None
    if ts_path.exists():
        print(f"\n[BENCHMARK] Profiling TorchScript CPU inference over {num_frames} frames...")
        ts_model = YOLO(str(ts_path))
        ts_latencies = []
        for _ in range(num_frames):
            t0 = time.time()
            ts_model.predict(dummy_frame, imgsz=imgsz, device="cpu", verbose=False)
            ts_latencies.append((time.time() - t0) * 1000.0)
        ts_avg_latency = sum(ts_latencies) / len(ts_latencies)
        ts_fps = 1000.0 / ts_avg_latency if ts_avg_latency > 0 else 0.0
        ts_p95 = np.percentile(ts_latencies, 95)
        ts_results = {"avg_latency": ts_avg_latency, "fps": ts_fps, "p95": ts_p95}

    process = psutil.Process(os.getpid())
    end_cpu_mem = process.memory_info().rss / (1024 ** 2)

    print("\n" + "=" * 65)
    print("  CPU INFERENCE BENCHMARK RESULTS")
    print("=" * 65)
    print(f"  PyTorch CPU Throughput:       {fps:.2f} FPS (Latency: {avg_latency:.2f} ms, P95: {p95_latency:.2f} ms)")
    if ts_results:
        print(f"  TorchScript CPU Throughput:   {ts_results['fps']:.2f} FPS (Latency: {ts_results['avg_latency']:.2f} ms, P95: {ts_results['p95']:.2f} ms)")
    print(f"  Process Memory RSS:           {end_cpu_mem:.2f} MB")
    print(f"  CPU Utilization:              {psutil.cpu_percent():.1f}%")
    print("=" * 65)

    return {
        "engine": "PyTorch CPU",
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "memory_mb": round(end_cpu_mem, 2)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/yolo/yolo11n.pt")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()

    benchmark_inference(args.weights, args.frames, args.imgsz)
