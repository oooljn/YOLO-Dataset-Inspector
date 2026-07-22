from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_label_file(label_path: Path) -> tuple[int, list[dict[str, Any]]]:
    """
    解析并检查一个 YOLO 标签文件。

    Returns:
        annotation_count: 标签行数量
        errors: 错误信息列表
    """
    errors: list[dict[str, Any]] = []
    annotation_count = 0

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return 0, [
            {
                "file": str(label_path),
                "line": None,
                "message": f"无法读取标签文件：{exc}",
            }
        ]

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line:
            continue

        annotation_count += 1
        fields = line.split()

        if len(fields) != 5:
            errors.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "message": "标签字段数量不是 5",
                }
            )
            continue

        class_text, x_text, y_text, width_text, height_text = fields

        try:
            class_value = float(class_text)
            x_center = float(x_text)
            y_center = float(y_text)
            width = float(width_text)
            height = float(height_text)
        except ValueError:
            errors.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "message": "标签中包含非数字字段",
                }
            )
            continue

        if not class_value.is_integer() or class_value < 0:
            errors.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "message": "类别编号必须是非负整数",
                }
            )

        coordinates = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }

        for name, value in coordinates.items():
            if not 0 <= value <= 1:
                errors.append(
                    {
                        "file": str(label_path),
                        "line": line_number,
                        "message": f"{name}={value} 不在 [0, 1] 范围内",
                    }
                )

        if width <= 0 or height <= 0:
            errors.append(
                {
                    "file": str(label_path),
                    "line": line_number,
                    "message": "边界框宽度和高度必须大于 0",
                }
            )

    return annotation_count, errors


def inspect_dataset(dataset_path: str) -> dict[str, Any]:
    """
    检查 YOLO 格式数据集。

    预期结构：
        dataset/
        ├── images/
        └── labels/
    """
    root = Path(dataset_path)
    images_dir = root / "images"
    labels_dir = root / "labels"

    result: dict[str, Any] = {
        "dataset_path": str(root),
        "image_count": 0,
        "label_count": 0,
        "annotation_count": 0,
        "missing_labels": [],
        "orphan_labels": [],
        "empty_labels": [],
        "invalid_annotations": [],
        "errors": [],
    }

    if not root.exists():
        result["errors"].append("数据集目录不存在")
        return result

    if not images_dir.is_dir():
        result["errors"].append("images 目录不存在")

    if not labels_dir.is_dir():
        result["errors"].append("labels 目录不存在")

    if result["errors"]:
        return result

    image_files = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    label_files = [
        path
        for path in labels_dir.glob("*.txt")
        if path.is_file()
    ]

    result["image_count"] = len(image_files)
    result["label_count"] = len(label_files)

    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    result["missing_labels"] = sorted(image_stems - label_stems)
    result["orphan_labels"] = sorted(label_stems - image_stems)

    for label_path in label_files:
        if label_path.stat().st_size == 0:
            result["empty_labels"].append(label_path.name)
            continue

        count, errors = parse_label_file(label_path)
        result["annotation_count"] += count
        result["invalid_annotations"].extend(errors)

    return result


if __name__ == "__main__":
    dataset = input("请输入 YOLO 数据集路径：").strip()
    report = inspect_dataset(dataset)
    print(report)
