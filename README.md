# EasyFind: Intelligent Real-Time Missing Child Detection & Child Welfare Management System

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/hardware-STRICT%20CPU%20ONLY-green.svg)]()
[![AI Verification](https://img.shields.io/badge/Featherless.ai-Qwen3.8--Flash--Next-purple.svg)]()
[![Security](https://img.shields.io/badge/encryption-AES--256--GCM-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

EasyFind is a comprehensive, production-grade, CPU-optimized police authority and child welfare management platform. Built for municipal police departments, transit authorities, and child protection agencies, EasyFind operates strictly on standard enterprise CPU hardware without requiring expensive GPU clusters. The platform delivers end-to-end missing child identification: from live CCTV surveillance and forensic video upload analysis to automated multimodal AI verification via Featherless.ai, cross-camera trajectory tracking, age-progression synthesis, and secure legal reunification workflows.

---

## Introduction & Problem Statement

In bustling urban environments, transit hubs (railway stations, metro terminals, bus depots, airports), and crowded public events, thousands of children are reported missing annually. The initial 1 to 3 hours following a disappearance—known critically as the **"Golden Window"**—are vital for interception and recovery before transit out of municipal jurisdiction occurs.

### Key Challenges in Conventional Systems
- **Surveillance Overload:** Municipal command centers monitor hundreds of concurrent camera feeds. Manual inspection is slow, cognitively exhausting, and inherently error-prone.
- **Hardware Bottlenecks:** Most state-of-the-art vision models demand high-end GPUs with high power consumption, making decentralized deployment across local precinct workstations cost-prohibitive.
- **Siloed CCTV Nodes:** Standard systems evaluate cameras independently, unable to track a subject traversing across disjoint physical camera checkpoints.
- **Biometric Degradation in Cold Cases:** When children remain missing for extended periods, standard facial recognition models fail due to facial morphological growth and age changes.
- **Privacy & Security Vulnerabilities:** Unencrypted biometric databases and missing child photos risk exfiltration or forensic tampering, creating severe legal and ethical liabilities.

### The EasyFind Solution
EasyFind resolves these critical challenges through a unified, CPU-native architecture that pairs edge computer vision (YOLO11 + BoT-SORT) with automated multimodal vision verification powered by **Featherless.ai** (`Qwen/Qwen3.8-Flash-Next`), instant zero-retraining vector enrollment, predictive age progression, and authenticated AES-256-GCM / SHA-3-256 cryptographic security.

---

## Automated AI Verification with Featherless.ai

As demanded by hackathon criteria, EasyFind integrates automated AI-driven forensic verification to eliminate reliance on purely manual inspection bottlenecks while maintaining officer-in-the-loop oversight.

- **Multimodal Vision Model:** Integrated with **Featherless.ai** using the state-of-the-art **`Qwen/Qwen3.8-Flash-Next`** multimodal vision-language model.
- **Automated Forensic Evaluation:** When local edge biometrics (AdaFace/iResNet 512-D) detect a candidate match above the operational threshold, the system autonomously submits the registered case photograph alongside the high-resolution CCTV facial crop to Featherless.ai.
- **LLM Reasoning & Risk Scoring:** The model conducts a deep visual alignment analysis—evaluating facial geometry, ocular distance, distinctive marks, clothing/contextual features, and environmental lighting discrepancies.
- **Actionable Decision Classification:** Featherless.ai outputs an automated verdict (`Confirmed Match`, `Requires Escalation`, or `False Positive Reject`), confidence assessment (`High`, `Medium`, `Low`), detailed forensic reasoning, and recommended police operational procedures.
- **Officer-in-the-Loop Oversight:** Police officers review the automated AI assessment directly within the verification portal and retain the ability to override verdicts or record field dispatch remarks.

---

## 4 Core Pillar Features

### 1. Cross-Camera Detection & Trajectory Reconstruction
- **Multi-Camera Re-ID:** Employs BoT-SORT and ByteTrack multi-object tracking combined with deep person Re-Identification feature extractors to link sightings across disjoint camera locations.
- **Spatio-Temporal Association:** Enforces geographic transit constraints (`CAM-01 -> CAM-02 -> CAM-03`) to eliminate physically impossible transit matches and validate plausible movement vectors.
- **Interactive Trajectory Mapping:** Chronological sighting sequences and transit routes are plotted automatically on an interactive Leaflet.js map, displaying waypoint confidence, timestamps, and movement vectors for police intercept units.

### 2. Live CCTV Streams & Forensic Footage Uploads
- **Authorized RTSP CCTV Integration:** Accepts live RTSP camera feeds (`rtsp://user:pass@ip:port/stream`) from municipal CCTV installations with automated URL validation, heartbeat monitoring, and failover demo simulation mode.
- **Forensic Video Batch Ingestion:** Enables investigators to upload external surveillance recordings (MP4, AVI, MKV, MOV) from station kiosks, private shops, or buses.
- **Automated Keyframe Extraction:** Scans video footage frame-by-frame on CPU, filters candidate persons through a biometric quality gate, and generates timestamped forensic keyframe galleries with one-click escalation to the verification portal.

### 3. Age Progression Modeling
- **Auxiliary Hypothesis Synthesis:** Built-in `AgeProgressionModel` simulates human facial growth trajectories (+2 years, +5 years, and +10 years) for cold cases and long-term missing children.
- **Morphological Feature Transforms:** Models adolescent cranial expansion, jaw lengthening, dermal texture maturity, and facial proportions to construct auxiliary reference projections.
- **Investigatory Enrolment:** Synthesized projections augment the memory vector index, enabling the matching engine to flag children whose physical appearance has matured significantly since the original photograph was captured.

### 4. Military-Grade Database & Biometric Encryption
- **AES-256-GCM Authenticated Encryption:** All sensitive facial photographs, CCTV crop snapshots, and biometric templates are encrypted at rest using AES-256-GCM with unique 96-bit initialization vectors (IV) to prevent unauthorized exfiltration.
- **SHA-3-256 Integrity Verification:** Every case file, biometric enrollment, and officer status update is cryptographically hashed with SHA-3-256. Any tamper attempt invalidates the chain-of-custody signature.
- **Audit-Sealed Chronology:** Every operational action, status transition, and officer login is recorded in an immutable, tamper-evident audit ledger for legal admissibility in court proceedings.

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
    │                                   ▼
    │                     ┌───────────────────────────────┐
    │                     │ Featherless.ai Vision LLM     │
    │                     │ (Qwen/Qwen3.8-Flash-Next)     │
    │                     │ Automated Forensic Assessment │
    │                     └───────────────┬───────────────┘
    │                                     │
    ▼                                     ▼
Cross-Camera Re-ID Engine ──► Officer Verification Dashboard (Side-by-Side Review)
                                          │
                                          ▼
                            Recovery & Child Welfare Lifecycle
```

---

## Quickstart & Installation

### 1. Requirements
- **OS:** Windows 10/11, Linux (Ubuntu/Debian), or macOS
- **Python:** 3.10 to 3.13
- **RAM:** 8GB minimum (16GB recommended)
- **Hardware:** Standard CPU (No GPU required)
- **Featherless.ai API Key:** Set via `FEATHERLESS_API_KEY` environment variable

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Optional: customize Featherless API key and model
export FEATHERLESS_API_KEY="your-featherless-api-key"
export FEATHERLESS_MODEL="Qwen/Qwen3.8-Flash-Next"
```

### 4. Run the Web Application
```bash
python run.py
```
Open your browser at: **`http://127.0.0.1:5000`**

**Default Officer Credentials:**
- **Username:** `officer`
- **Password:** `police123`

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

Run the complete acceptance test suite and cryptography validation:
```bash
python -m pytest tests/test_crypto.py -v
python -m pytest tests/test_cpu_acceptance.py -v -s
python -m pytest tests/test_featherless.py -v -k "not test_featherless_live_assessment"
python -m pytest tests/test_rtsp_stream.py -v
```

All acceptance tests run natively on CPU and output actual measured metrics to `tests/cpu_acceptance_results.json`.

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
