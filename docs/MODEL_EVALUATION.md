# Model Evaluation & Performance Target Analysis

## 1. Evaluation Protocol
The system evaluates detection, tracking, Re-ID, and face recognition independently.

### A. Person Detection Metrics (Held-Out Test Set)
Evaluated using `python training/yolo/evaluate_yolo.py`:
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall:** $\frac{TP}{TP + FN}$
- **mAP@50:** Mean Average Precision at IoU threshold 0.50
- **mAP@50-95:** Mean Average Precision averaged from IoU 0.50 to 0.95
- **F1 Score:** $2 \cdot \frac{P \cdot R}{P + R}$

### Actual Measured Results (CCTV Test Split):
| Metric | Measured Value | Performance vs 96% Target |
|---|---|---|
| Precision | 1.17% | Needs Improvement (High false alarm on un-annotated background) |
| Recall | 94.59% | Close to target (94.6% vs 96.0%) |
| mAP@50 | 49.90% | Domain adapted baseline |
| mAP@50-95 | 16.52% | Baseline |

### B. Person Re-ID Metrics (CMC Curve)
Evaluated using `python training/reid/evaluate_reid.py`:
- **Rank-1 Accuracy:** 100.0%
- **Rank-5 Accuracy:** 100.0%
- **mAP:** 100.0%

### C. Biometric Face Verification
- **Embedding Dimension:** 512-D L2-normalized
- **Verification Cosine Threshold:** 0.62 (62%)
- **Vector Search Latency:** 0.104 ms on CPU
