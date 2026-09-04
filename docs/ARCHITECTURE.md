# System Architecture Documentation

## 1. Executive Summary
Hackwave is architected as a modular, CPU-optimized police surveillance and child welfare platform. The architecture separates offline model training from real-time stream ingestion and separates AI inference from the Flask HTTP request cycle.

## 2. Structural Decomposition

```
Hackwave/
├── app/                  # Application Core
│   ├── ai/               # Standalone AI/CV Engine (Independent of web layer)
│   │   ├── detection/    # YOLO11n CPU Person Detector
│   │   ├── tracking/     # BoT-SORT / ByteTrack CPU Multi-Object Tracking
│   │   ├── face/         # YuNet Detector, Quality Gate, AdaFace / iResNet
│   │   ├── reid/         # Person Re-ID Feature Extractor & Topology Graph
│   │   ├── enhancement/  # Selective Super-Resolution & Unsharp Masking
│   │   ├── age_progression/# Modular Age Progression Reference Generator
│   │   └── pipeline/     # Asynchronous Stream Workers & Worker Pool
│   ├── database/         # SQLAlchemy ORM (SQLite / PostgreSQL with pgvector readiness)
│   ├── security/         # AES-256-GCM & SHA-3-256 Authenticated Encryption
│   ├── services/         # Business Logic (Case, Alert, Movement, Welfare)
│   ├── routes/           # REST APIs & View Blueprints
│   └── templates/        # 10 Police Authority UI Views
├── training/             # Offline Model Training & Evaluation Subsystem
│   ├── yolo/             # Dataset prep, validation, training, eval, export
│   └── reid/             # Re-ID Triplet Loss fine-tuning & CMC evaluation
├── models/               # Checkpoints and serialized weights
├── datasets/             # Annotated benchmark datasets
└── tests/                # Automated unit and acceptance test suites
```

## 3. Computational Design Principles
- **Decoupled Asynchronous Workers:** Flask HTTP request threads never perform computer vision operations. Video streams and RTSP feeds are decoded and processed in background worker threads (`StreamWorker`).
- **Periodic Face Analysis:** Face recognition is computationally heavy on CPU (~68ms per face). BoT-SORT tracks individuals persistently across frames, and face recognition is triggered only once every $N$ frames per track (e.g. $N=10$).
- **In-Memory Vector Search:** Case embeddings are pre-loaded into an in-memory dictionary. Biometric similarity comparison is performed using vectorized dot products in $0.1$ milliseconds without querying the SQL database.
