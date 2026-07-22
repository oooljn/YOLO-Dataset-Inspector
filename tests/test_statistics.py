from pathlib import Path

from PIL import Image

from src.statistics import analyze_dataset


def test_dataset_statistics(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    image = Image.new("RGB", (640, 480), "white")
    image.save(images_dir / "image001.jpg")

    (labels_dir / "image001.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n"
        "1 0.4 0.4 0.5 0.5\n",
        encoding="utf-8",
    )

    (tmp_path / "data.yaml").write_text(
        "names:\n"
        "  0: person\n"
        "  1: car\n",
        encoding="utf-8",
    )

    result = analyze_dataset(str(tmp_path))

    assert result["image_count"] == 1
    assert result["label_count"] == 1
    assert result["total_objects"] == 2
    assert result["class_counts"][0] == 1
    assert result["class_counts"][1] == 1
    assert result["image_sizes"]["640x480"] == 1
    assert result["class_names"][0] == "person"
    assert result["class_names"][1] == "car"


def test_statistics_nonexistent_dataset() -> None:
    result = analyze_dataset(
        "this_dataset_does_not_exist"
    )

    assert result["image_count"] == 0
    assert result["total_objects"] == 0
    assert result["errors"]
