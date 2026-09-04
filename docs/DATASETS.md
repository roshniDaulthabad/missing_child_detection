# Dataset Strategy & Validation Protocol

## 1. Dataset Distinctions
- **Training Dataset (Pedestrian Tracking Benchmark):** Used strictly for initializing YOLO11 person detection and tracker weights.
- **Biometric Application Database:** Contains real registered missing child case data, photos, and embeddings. The AI model is NEVER retrained on case data.

## 2. Dataset Preparation
To generate or convert the CCTV pedestrian dataset:
```bash
python training/yolo/prepare_dataset.py
```
Output structure:
```
datasets/cctv_pedestrians/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## 3. Dataset Validation Checks
Run `training/yolo/validate_dataset.py` before any training run to check:
1. File corruption (unreadable images)
2. Label format (5 values per line: `<class> <xc> <yc> <w> <h>`)
3. Class ID validity (strictly Class 0 for pedestrians)
4. Coordinate range bounds ($[0.0, 1.0]$)
5. Split leakage: SHA-256 hash comparison across train, val, and test splits
