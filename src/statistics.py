"""
YOLO dataset statistics analysis module.

Analyze a YOLO-format dataset and return:
- class distribution counts
- image size distribution
- total object count
"""

import os
from pathlib import Path

import yaml
from PIL import Image


def analyze_dataset(dataset_path: str) -> dict:
    """
    Analyze YOLO dataset statistics.

    Args:
        dataset_path: Path to the dataset directory containing
                      images/, labels/, and data.yaml.

    Returns:
        dict with keys:
            - class_counts: dict mapping class_id to count
            - class_names: dict mapping class_id to name (from data.yaml)
            - image_sizes: dict mapping "WxH" to count
            - total_objects: total number of annotated objects
            - image_count: total number of images
            - label_count: total number of label files
    """
    dataset = Path(dataset_path)

    # Parse data.yaml
    yaml_path = dataset / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        names = yaml_data.get("names", {})
    else:
        names = {}

    # Count class distribution
    class_counts = {}
    total_objects = 0
    label_count = 0

    labels_dir = dataset / "labels"
    if labels_dir.exists():
        for label_file in labels_dir.glob("*.txt"):
            label_count += 1
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_objects += 1

    # Count image sizes
    image_sizes = {}
    image_count = 0

    images_dir = dataset / "images"
    if images_dir.exists():
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                try:
                    with Image.open(img_file) as img:
                        w, h = img.size
                        size_key = f"{w}x{h}"
                        image_sizes[size_key] = image_sizes.get(size_key, 0) + 1
                    image_count += 1
                except Exception:
                    pass

    return {
        "class_counts": class_counts,
        "class_names": names,
        "image_sizes": image_sizes,
        "total_objects": total_objects,
        "image_count": image_count,
        "label_count": label_count,
    }
