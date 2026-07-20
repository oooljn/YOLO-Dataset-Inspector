"""
YOLO dataset statistics analysis module.

Analyze a YOLO-format dataset and return:
- class distribution counts
- image size distribution
- total object count
"""

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
            - errors: list of error/warning messages
    """
    dataset = Path(dataset_path)
    errors = []

    # Validate dataset directory exists
    if not dataset.exists():
        return {
            "class_counts": {},
            "class_names": {},
            "image_sizes": {},
            "total_objects": 0,
            "image_count": 0,
            "label_count": 0,
            "errors": [f"Dataset path not found: {dataset_path}"],
        }

    # Parse data.yaml
    yaml_path = dataset / "data.yaml"
    names = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
            if yaml_data and isinstance(yaml_data, dict):
                names = yaml_data.get("names", {})
            else:
                errors.append("data.yaml is empty or invalid")
        except yaml.YAMLError as e:
            errors.append(f"Failed to parse data.yaml: {e}")
    else:
        errors.append("data.yaml not found in dataset directory")

    # Count class distribution
    class_counts = {}
    total_objects = 0
    label_count = 0
    invalid_lines = 0

    labels_dir = dataset / "labels"
    if labels_dir.exists():
        for label_file in sorted(labels_dir.glob("*.txt")):
            label_count += 1
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            invalid_lines += 1
                            continue
                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            invalid_lines += 1
                            continue
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_objects += 1
            except (IOError, OSError) as e:
                errors.append(f"Cannot read {label_file.name}: {e}")
    else:
        errors.append("labels/ directory not found")

    if invalid_lines > 0:
        errors.append(f"Skipped {invalid_lines} invalid annotation lines")

    # Count image sizes
    image_sizes = {}
    image_count = 0

    images_dir = dataset / "images"
    if images_dir.exists():
        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                try:
                    with Image.open(img_file) as img:
                        w, h = img.size
                        size_key = f"{w}x{h}"
                        image_sizes[size_key] = image_sizes.get(size_key, 0) + 1
                    image_count += 1
                except Exception:
                    errors.append(f"Cannot read image: {img_file.name}")
    else:
        errors.append("images/ directory not found")

    return {
        "class_counts": class_counts,
        "class_names": names,
        "image_sizes": image_sizes,
        "total_objects": total_objects,
        "image_count": image_count,
        "label_count": label_count,
        "errors": errors,
    }


def print_statistics(stats: dict) -> None:
    """
    Print formatted statistics report to console.

    Args:
        stats: The dict returned by analyze_dataset().
    """
    print("=" * 50)
    print("YOLO Dataset Statistics Report")
    print("=" * 50)
    print(f"Images:  {stats['image_count']}")
    print(f"Labels:  {stats['label_count']}")
    print(f"Objects: {stats['total_objects']}")
    print()

    if stats["class_names"]:
        print("Class mapping (from data.yaml):")
        for cid, cname in stats["class_names"].items():
            count = stats["class_counts"].get(cid, 0)
            print(f"  {cid}: {cname} ({count} objects)")
    else:
        print("Class distribution:")
        for cid, count in sorted(stats["class_counts"].items()):
            print(f"  class_{cid}: {count} objects")

    print()
    print("Image size distribution:")
    for size, count in sorted(stats["image_sizes"].items()):
        print(f"  {size}: {count} images")

    if stats["errors"]:
        print()
        print("Issues found:")
        for err in stats["errors"]:
            print(f"  ⚠ {err}")
    print("=" * 50)
