# Model Training & Fine-Tuning Guide

## 1. Hardware Detection & Constraints
The training scripts explicitly enforce `device="cpu"` and print the CPU computational profile banner prior to training:
```bash
python training/yolo/train_yolo.py --epochs 3 --imgsz 416 --batch 4 --device cpu
```

## 2. CCTV Domain Augmentation
Surveillance cameras operate at high angles, with variable lighting, compression artifacts, and low resolution. Augmentations configured in `train_yolo.py`:
- `hsv_v=0.4`: Simulates day-to-night lighting and under-exposure
- `hsv_s=0.5`: Simulates camera color distortion
- `degrees=5.0`: Subtle camera tilt angles
- `scale=0.3`: Varying person distance from camera
- `shear=2.0`: Perspective foreshortening
- `mosaic=0.5`: Multi-person crowd context
- `mixup=0.0`: Disabled to maintain realistic CCTV structure

## 3. Person Re-ID Training
Person Re-ID is trained independently of detection using Triplet Loss:
$$\mathcal{L}_{triplet} = \max(0, \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + \alpha)$$
where $a$ is anchor, $p$ is positive (same identity, different camera), $n$ is negative (different identity), and $\alpha=0.3$ is the margin.
```bash
python training/reid/train_reid.py --epochs 3 --batch 4
```
