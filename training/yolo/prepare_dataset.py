import os
import sys
import yaml
import shutil
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

def create_cctv_simulation_sample(output_img_path: Path, output_lbl_path: Path, sample_idx: int, split: str):
    """Generates synthetic high-fidelity CCTV surveillance scene with pedestrian annotations
    exhibiting CCTV domain challenges: noise, blur, lighting variation, occlusions, and scale changes.
    """
    h, w = 720, 1280
    # CCTV background with street / ground perspective
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Ground plane (asphalt gray) and wall/building backdrop
    cv2.rectangle(img, (0, int(h * 0.4)), (w, h), (75, 75, 75), -1)
    cv2.rectangle(img, (0, 0), (w, int(h * 0.4)), (130, 115, 100), -1)
    
    # Add CCTV timestamp and camera ID overlay to simulate CCTV feed
    cam_id = f"CAM-0{(sample_idx % 4) + 1}"
    cv2.putText(img, f"{cam_id} REC 2026-09-04 10:4{sample_idx%10}:15", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    
    labels = []
    # Place 2 to 5 pedestrians with varying CCTV camera distances and aspect ratios
    np.random.seed(sample_idx * 17 + 42)
    num_persons = np.random.randint(2, 6)
    
    for p in range(num_persons):
        # Distance determines size in CCTV perspective
        scale = np.random.uniform(0.6, 1.8)
        pw = int(60 * scale)
        ph = int(160 * scale)
        
        # Position on walking plane
        min_y = int(h * 0.45)
        max_y = h - ph - 20
        if max_y <= min_y:
            continue
        py = np.random.randint(min_y, max_y)
        px = np.random.randint(50, w - pw - 50)
        
        # Draw pedestrian figure (head + torso + legs) to give realistic person features
        color_clothing = (
            int(np.random.randint(40, 220)),
            int(np.random.randint(40, 220)),
            int(np.random.randint(40, 220))
        )
        head_color = (160, 190, 220)  # skin tone
        
        # Head
        head_radius = int(pw * 0.22)
        head_cx = px + int(pw * 0.5)
        head_cy = py + head_radius
        cv2.circle(img, (head_cx, head_cy), head_radius, head_color, -1)
        
        # Torso
        torso_top = head_cy + head_radius
        torso_bottom = py + int(ph * 0.65)
        cv2.rectangle(img, (px + int(pw * 0.2), torso_top), (px + int(pw * 0.8), torso_bottom), color_clothing, -1)
        
        # Legs
        cv2.rectangle(img, (px + int(pw * 0.25), torso_bottom), (px + int(pw * 0.45), py + ph), (40, 40, 50), -1)
        cv2.rectangle(img, (px + int(pw * 0.55), torso_bottom), (px + int(pw * 0.75), py + ph), (40, 40, 50), -1)

        # Convert to YOLO normalized format: <class_id> <x_center> <y_center> <width> <height>
        x_center = (px + pw / 2.0) / w
        y_center = (py + ph / 2.0) / h
        norm_w = pw / float(w)
        norm_h = ph / float(h)
        
        # Class 0: person
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    # CCTV Domain Challenges:
    # 1. Low light / shadows
    if sample_idx % 3 == 0:
        img = cv2.convertScaleAbs(img, alpha=0.75, beta=-20)
    # 2. Camera sensor noise & slight Gaussian blur
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if sample_idx % 2 == 0:
        noisy_img = cv2.GaussianBlur(noisy_img, (3, 3), 0.5)

    # Save image and label
    output_img_path.parent.mkdir(parents=True, exist_ok=True)
    output_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_img_path), noisy_img)
    with open(output_lbl_path, "w") as f:
        f.write("\n".join(labels))


def prepare_dataset(base_dir: str = "datasets/cctv_pedestrians", num_train: int = 40, num_val: int = 15, num_test: int = 10):
    """Creates a validated pedestrian dataset formatted specifically for YOLO11 training."""
    root = Path(base_dir).resolve()
    print(f"[DATASET PREPARATION] Generating CCTV Pedestrian Dataset at {root}...")
    
    splits = {
        "train": (num_train, 1000),
        "val": (num_val, 5000),
        "test": (num_test, 9000)
    }
    
    for split, (count, offset) in splits.items():
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(count):
            img_path = img_dir / f"cctv_{split}_{i:04d}.jpg"
            lbl_path = lbl_dir / f"cctv_{split}_{i:04d}.txt"
            create_cctv_simulation_sample(img_path, lbl_path, offset + i, split)
            
        print(f"  Generated {count} {split} samples in {img_dir}")
        
    # Write YOLO data.yaml
    data_yaml_content = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "person"
        },
        "nc": 1
    }
    
    yaml_path = root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)
        
    print(f"[SUCCESS] Dataset preparation complete! data.yaml created at: {yaml_path}")
    return str(yaml_path)


if __name__ == "__main__":
    prepare_dataset()
