# AI & Computer Vision Pipeline

## 1. Pipeline Stages

```
Input Video Frame (1280x720)
       │
       ▼ [Stage 1: Frame Preprocessing]
Resize to Mode Imgsz (416 / 512 / 640)
       │
       ▼ [Stage 2: Person Detection]
YOLO11n CPU Inference (Class 0: Person)
       │
       ▼ [Stage 3: Multi-Object Tracking]
BoT-SORT / ByteTrack Tracking (Persistent Track IDs)
       │
       ▼ [Stage 4: Scheduling & Quality Gate]
Track Due for Biometrics? (Frames since check >= N)
  ├─ No  ──► Update Track Trajectory Trail
  └─ Yes ──► Face Detection (YuNet ONNX / CCTV Head-Region Fallback)
                 │
                 ▼ [Stage 5: Quality Assessment]
             Assess Sharpness (Laplacian Var >= 35) & Size (>= 32px)
                 │
                 ▼ [Stage 6: Selective Enhancement]
             If low resolution: Apply Lanczos-CLAHE Enhancer
                 │
                 ▼ [Stage 7: AdaFace Biometric Embedding]
             Preprocess to 112x112 RGB Normalized [-1, 1]
             iResNet100 Forward Pass ──► 512-D L2-Normalized Vector
                 │
                 ▼ [Stage 8: Similarity Search]
             Cosine Dot Product vs In-Memory Active Embeddings Cache
                 │
                 ▼ [Stage 9: Threshold Filtering]
             Similarity Score >= 0.62 (62%)?
                 ├─ Yes ──► Generate Potential Match Alert
                 └─ No  ──► Register Re-ID Tracklet for Cross-Camera Links
```

## 2. Distinction of Core Problems
1. **Detection (YOLO11):** Locates person bounding boxes in raw CCTV scenes.
2. **Tracking (BoT-SORT):** Associates bounding boxes frame-to-frame within a single camera.
3. **Face Recognition (AdaFace / iResNet):** Compares facial structure against registered missing child cases.
4. **Person Re-ID:** Matches body appearance across cameras when face is unobserved or low-resolution.
5. **Human Verification:** Authorized officer confirms or rejects matches before legal action.
