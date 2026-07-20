"""
YOLO dataset statistics analysis module.

Analyze a YOLO-format dataset and return:
- class distribution counts
- image size distribution
- total object count
"""

from pathlib import Path
from typing import Union

import yaml

# Lazy import: Pillow is only needed when actually loading images
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _normalize_class_names(raw_names: Union[list, dict, None]) -> dict:
    """
    Convert YOLO names (list or dict) into uniform {int: str} format.

    Examples:
        ['cat', 'dog']          -> {0: 'cat', 1: 'dog'}
        {0: 'cat', 1: 'dog'}    -> {0: 'cat', 1: 'dog'}
        {'0': 'cat', '1': 'dog'} -> {0: 'cat', 1: 'dog'}
    """
    if raw_names is None:
        return {}
    if isinstance(raw_names, list):
        return {i: str(name) for i, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    return {}


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
            - objects_per_image: dict with min, max, avg objects per image
            - bbox_area_dist: dict with small, medium, large bbox counts
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
            "objects_per_image": {"min": 0, "max": 0, "avg": 0.0},
            "bbox_area_dist": {"small": 0, "medium": 0, "large": 0},
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
                names = _normalize_class_names(yaml_data.get("names"))
            else:
                errors.append("data.yaml is empty or invalid")
        except yaml.YAMLError as e:
            errors.append(f"Failed to parse data.yaml: {e}")
    else:
        errors.append("data.yaml not found in dataset directory")

    # Count class distribution and per-image stats
    class_counts = {}
    bbox_area_dist = {"small": 0, "medium": 0, "large": 0}
    objects_per_file = []
    total_objects = 0
    label_count = 0
    invalid_lines = 0

    labels_dir = dataset / "labels"
    if labels_dir.exists():
        for label_file in sorted(labels_dir.glob("*.txt")):
            label_count += 1
            file_obj_count = 0
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            invalid_lines += 1
                            continue
                        try:
                            class_id = int(parts[0])
                            bbox_w = float(parts[3])
                            bbox_h = float(parts[4])
                        except ValueError:
                            invalid_lines += 1
                            continue
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_objects += 1
                        file_obj_count += 1

                        # Classify bbox area (normalized, 0-1)
                        area = bbox_w * bbox_h
                        if area < 0.01:
                            bbox_area_dist["small"] += 1
                        elif area < 0.10:
                            bbox_area_dist["medium"] += 1
                        else:
                            bbox_area_dist["large"] += 1
            except (IOError, OSError) as e:
                errors.append(f"Cannot read {label_file.name}: {e}")
            objects_per_file.append(file_obj_count)
    else:
        errors.append("labels/ directory not found")

    if invalid_lines > 0:
        errors.append(f"Skipped {invalid_lines} invalid annotation lines")

    # Compute objects-per-image statistics
    if objects_per_file:
        objects_per_image = {
            "min": min(objects_per_file),
            "max": max(objects_per_file),
            "avg": round(sum(objects_per_file) / len(objects_per_file), 1),
        }
    else:
        objects_per_image = {"min": 0, "max": 0, "avg": 0.0}

    # Count image sizes
    image_sizes = {}
    image_count = 0

    images_dir = dataset / "images"
    if images_dir.exists():
        if not _HAS_PIL:
            errors.append("Pillow not installed — image size statistics unavailable")
        else:
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
        "objects_per_image": objects_per_image,
        "bbox_area_dist": bbox_area_dist,
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

    opp = stats.get("objects_per_image", {})
    if opp:
        print(f"Objects per image:  min={opp['min']}  max={opp['max']}  avg={opp['avg']}")

    bbox = stats.get("bbox_area_dist", {})
    if bbox:
        print(f"BBox area distribution:  small={bbox['small']}  medium={bbox['medium']}  large={bbox['large']}")
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
            print(f"  [!] {err}")
    print("=" * 50)
