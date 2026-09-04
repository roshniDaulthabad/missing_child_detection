import os
import sys
import yaml
import cv2
import hashlib
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

class DatasetValidator:
    """Rigorous dataset validation ensuring zero corruption, no label mismatch, and no split leakage."""

    def __init__(self, data_yaml_path: str):
        self.yaml_path = Path(data_yaml_path).resolve()
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at: {self.yaml_path}")
            
        with open(self.yaml_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.root = Path(self.cfg.get("path", self.yaml_path.parent))
        self.splits = ["train", "val", "test"]
        self.report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
            "hashes": defaultdict(dict)
        }

    def validate(self) -> dict:
        print("=" * 65)
        print("  DATASET VALIDATION & INTEGRITY INSPECTION")
        print("=" * 65)
        print(f"  Configuration: {self.yaml_path}")
        print(f"  Root Path:     {self.root}")

        total_images = 0
        total_boxes = 0

        for split in self.splits:
            split_rel = self.cfg.get(split)
            if not split_rel:
                self.report["warnings"].append(f"Split '{split}' not defined in data.yaml")
                continue
                
            img_dir = self.root / split_rel
            if not img_dir.exists():
                self.report["errors"].append(f"Image directory does not exist: {img_dir}")
                self.report["valid"] = False
                continue

            lbl_dir = self.root / "labels" / split
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
            
            split_img_count = len(images)
            split_box_count = 0
            
            for img_path in images:
                # 1. Image corruption check
                try:
                    mat = cv2.imread(str(img_path))
                    if mat is None:
                        self.report["errors"].append(f"Corrupted image file: {img_path}")
                        self.report["valid"] = False
                        continue
                except Exception as e:
                    self.report["errors"].append(f"Failed to read image {img_path}: {e}")
                    self.report["valid"] = False
                    continue

                # Compute hash for duplicate and leakage check
                with open(img_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                self.report["hashes"][split][img_path.name] = file_hash

                # 2. Matching label check
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    self.report["errors"].append(f"Missing label for image: {img_path.name}")
                    self.report["valid"] = False
                    continue

                # 3. Label format & class ID check
                with open(lbl_path, "r") as lf:
                    lines = [l.strip() for l in lf.readlines() if l.strip()]

                if not lines:
                    self.report["warnings"].append(f"Empty annotation file: {lbl_path.name}")

                for line_idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        self.report["errors"].append(
                            f"Invalid YOLO format in {lbl_path.name} line {line_idx+1}: '{line}'"
                        )
                        self.report["valid"] = False
                        continue

                    try:
                        cls_id = int(parts[0])
                        xc, yc, w, h = [float(x) for x in parts[1:]]
                    except ValueError:
                        self.report["errors"].append(
                            f"Non-numeric value in {lbl_path.name} line {line_idx+1}: '{line}'"
                        )
                        self.report["valid"] = False
                        continue

                    if cls_id != 0:
                        self.report["errors"].append(
                            f"Invalid class ID {cls_id} in {lbl_path.name}. Only class 0 (person) is valid."
                        )
                        self.report["valid"] = False

                    # Bound checks
                    for coord_name, val in [("xc", xc), ("yc", yc), ("w", w), ("h", h)]:
                        if not (0.0 <= val <= 1.0):
                            self.report["errors"].append(
                                f"Coordinate {coord_name}={val} out of range [0, 1] in {lbl_path.name}"
                            )
                            self.report["valid"] = False

                    split_box_count += 1

            total_images += split_img_count
            total_boxes += split_box_count
            self.report["stats"][split] = {
                "images": split_img_count,
                "boxes": split_box_count
            }

        # 4. Train / Val / Test Leakage Check
        train_hashes = set(self.report["hashes"]["train"].values())
        val_hashes = set(self.report["hashes"]["val"].values())
        test_hashes = set(self.report["hashes"]["test"].values())

        train_val_overlap = train_hashes.intersection(val_hashes)
        train_test_overlap = train_hashes.intersection(test_hashes)
        if train_val_overlap:
            self.report["errors"].append(f"Data leakage detected! {len(train_val_overlap)} samples overlap between Train and Val!")
            self.report["valid"] = False
        if train_test_overlap:
            self.report["errors"].append(f"Data leakage detected! {len(train_test_overlap)} samples overlap between Train and Test!")
            self.report["valid"] = False

        # Summary output
        print("-" * 65)
        for split, data in self.report["stats"].items():
            print(f"  {split.upper():<6} Split: {data['images']:>4} images | {data['boxes']:>5} pedestrian boxes")
        print(f"  TOTAL:       {total_images:>4} images | {total_boxes:>5} pedestrian boxes")
        print("-" * 65)
        
        if self.report["valid"]:
            print("  STATUS: [PASS] All dataset integrity checks succeeded. No leakage found.")
        else:
            print(f"  STATUS: [FAIL] Found {len(self.report['errors'])} errors.")
            for err in self.report["errors"][:5]:
                print(f"    - {err}")
        print("=" * 65)

        # Write report JSON
        report_file = self.root / "dataset_validation_report.yaml"
        with open(report_file, "w") as rf:
            yaml.dump(self.report, rf)

        return self.report

if __name__ == "__main__":
    yaml_target = sys.argv[1] if len(sys.argv) > 1 else "datasets/cctv_pedestrians/data.yaml"
    validator = DatasetValidator(yaml_target)
    validator.validate()
