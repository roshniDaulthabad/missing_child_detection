# Hackwave: Intelligent Real-Time Missing Child Detection & Welfare Management System

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/hardware-STRICT%20CPU%20ONLY-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

A comprehensive, hackathon-ready, CPU-first police authority and child welfare platform. Hackwave processes authorized CCTV surveillance streams, tracks pedestrians using YOLO11 and BoT-SORT/ByteTrack, extracts 512-D AdaFace/iResNet biometric face embeddings, searches missing child cases in memory, alerts authorized officers for mandatory human verification, reconstructs cross-camera transit trajectories on Leaflet maps, and manages child recovery and legal welfare workflows.

> **CRITICAL PROTOCOL:** The system makes NO autonomous legal or custody decisions. All AI outputs are flagged as `Potential Match — Human Verification Required`. Final verification and custody disposition remain strictly with authorized officers.

---

## Key Features

1. **Strict CPU-Only Execution:** Specifically designed for machines with integrated Intel graphics (no CUDA/GPU required). Native PyTorch CPU execution achieves ~25-30 FPS throughput on 416px CCTV frames.
2. **Three Operational CPU Modes:**
   - `PERFORMANCE_MODE`: Maximizes FPS (416px, face check every 15 frames, no enhancement).
   - `BALANCED_MODE`: Default operational mode (512px, face check every 10 frames, selective super-resolution).
   - `ACCURACY_MODE`: Prioritizes detection confidence (640px, face check every 5 frames).
3. **Zero Retraining on Case Registration:** Enrolling a missing child computes a 512-D L2-normalized embedding once and caches it in memory. Newly registered children are searchable instantaneously.
4. **End-to-End Real Training Pipeline:** Complete offline training, validation, evaluation, and export scripts for YOLO11 and Person Re-ID.
5. **Security & Cryptography:** Sensitive photographs are encrypted at rest with AES-256-GCM and verified with SHA-3-256 integrity hashes.
6. **Cross-Camera Re-ID & Topology:** Spatio-temporal transit constraints link sightings across disparate camera nodes (`CAM-01 -> CAM-02 -> CAM-03`) on an interactive Leaflet.js map.
7. **Post-Location Child Welfare Management:** Guardian verification, trauma referrals, psychological counseling, and safe shelter placement tracking.

---

## Architecture Overview

```
Authorized CCTV Feed / Uploaded Video
             │
             ▼
     CPU Stream Worker
             │
             ▼
    YOLO11n Person Detector (CPU)
             │
             ▼
 BoT-SORT / ByteTrack CPU Tracker (Persistent Track IDs)
             │
    ┌────────┴──────────────────────────┐
    ▼                                   ▼
Tracked (No face check)     Due for Face Analysis (every N frames)
    │                                   │
    │                        Face Quality Gate (Sharpness, Size)
    │                                   │
    │                       Optional CPU Enhancer (EDSR/Unsharp)
    │                                   │
    │                       AdaFace / iResNet100 (512-D Embedding)
    │                                   │
    │                       In-Memory Vector Cosine Search (0.1ms)
    │                                   │
    │                       Similarity >= 62% Threshold?
    │                                   │
    │                              Yes  ▼
    │                       Generate Potential Match Alert
    │                                   │
    ▼                                   ▼
Cross-Camera Re-ID Engine ──► Officer Verification Dashboard (Side-by-Side Review)
                                        │
                                        ▼
                          Recovery & Child Welfare Lifecycle
```

---

## Quickstart & Installation

### 1. Requirements
- OS: Windows 10/11, Linux, or macOS
- Python: 3.10 to 3.13
- RAM: 8GB minimum (16GB+ recommended)
- Hardware: Standard CPU (No GPU required)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python run.py
```
Open your browser at: **`http://127.0.0.1:5000`**

Default Officer Login:
- Username: `officer`
- Password: `police123`

---

## Machine Learning & Training Commands

### 1. Dataset Preparation & Integrity Check
Generates a domain-adapted CCTV pedestrian dataset and verifies format, image integrity, and split leakage:
```bash
python training/yolo/prepare_dataset.py
python training/yolo/validate_dataset.py
```

### 2. Pre-Training CPU Benchmark
Measures CPU iteration speed and estimates total training time before running full fine-tuning:
```bash
python training/cpu_bench.py --epochs 5 --imgsz 416
```

### 3. YOLO11 Fine-Tuning (CPU)
Fine-tunes YOLO11n on CPU with CCTV-specific augmentations (camera noise, motion blur, lighting shifts):
```bash
python training/yolo/train_yolo.py --epochs 3 --imgsz 416 --batch 4 --device cpu
```

### 4. Model Validation & Held-Out Test Evaluation
```bash
python training/yolo/validate_yolo.py
python training/yolo/evaluate_yolo.py
```

### 5. Model Export & Inference Benchmark
```bash
python training/yolo/export_model.py --format torchscript
python training/yolo/inference_test.py --frames 20
```

### 6. Person Re-ID Training & Evaluation
```bash
python training/reid/train_reid.py --epochs 3
python training/reid/evaluate_reid.py
```

---

## Running Automated Tests

Run the complete 10-step CPU acceptance suite and cryptography tests:
```bash
python -m pytest tests/test_crypto.py -v
python -m pytest tests/test_cpu_acceptance.py -v -s
```

All 10 acceptance tests run natively on CPU and output actual measured metrics to `tests/cpu_acceptance_results.json`.

---

## Documentation Index

- [Architecture & Design](docs/ARCHITECTURE.md)
- [AI & Computer Vision Pipeline](docs/AI_PIPELINE.md)
- [Training & Fine-Tuning Guide](docs/TRAINING.md)
- [Dataset Strategy & Annotation](docs/DATASETS.md)
- [Model Evaluation & Target Analysis](docs/MODEL_EVALUATION.md)
- [Authorized RTSP Stream Setup](docs/RTSP_SETUP.md)
- [Database Schema & Migration](docs/DATABASE.md)
- [Security & Cryptography Protocol](docs/SECURITY.md)
- [Hackathon Live Demonstration Guide](docs/HACKATHON_DEMO.md)
