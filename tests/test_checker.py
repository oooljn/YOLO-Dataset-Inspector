from pathlib import Path

from PIL import Image

from src.checker import inspect_dataset


def create_image(path: Path) -> None:
    """创建一张简单测试图片。"""
    image = Image.new("RGB", (100, 100), "white")
    image.save(path)


def test_valid_dataset(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    create_image(images_dir / "image001.jpg")

    (labels_dir / "image001.txt").write_text(
        "0 0.5 0.5 0.3 0.2\n",
        encoding="utf-8",
    )

    result = inspect_dataset(str(tmp_path))

    assert result["image_count"] == 1
    assert result["label_count"] == 1
    assert result["annotation_count"] == 1
    assert result["missing_labels"] == []
    assert result["orphan_labels"] == []
    assert result["invalid_annotations"] == []


def test_missing_label(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    create_image(images_dir / "image001.jpg")

    result = inspect_dataset(str(tmp_path))

    assert "image001" in result["missing_labels"]


def test_orphan_label(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    (labels_dir / "orphan.txt").write_text(
        "0 0.5 0.5 0.3 0.2\n",
        encoding="utf-8",
    )

    result = inspect_dataset(str(tmp_path))

    assert "orphan" in result["orphan_labels"]


def test_empty_label(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    create_image(images_dir / "image001.jpg")
    (labels_dir / "image001.txt").write_text(
        "",
        encoding="utf-8",
    )

    result = inspect_dataset(str(tmp_path))

    assert "image001.txt" in result["empty_labels"]


def test_invalid_annotation_format(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    create_image(images_dir / "image001.jpg")

    (labels_dir / "image001.txt").write_text(
        "0 0.5 0.5\n",
        encoding="utf-8",
    )

    result = inspect_dataset(str(tmp_path))

    assert len(result["invalid_annotations"]) == 1


def test_out_of_range_annotation(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    images_dir.mkdir()
    labels_dir.mkdir()

    create_image(images_dir / "image001.jpg")

    (labels_dir / "image001.txt").write_text(
        "0 1.2 0.5 0.3 0.2\n",
        encoding="utf-8",
    )

    result = inspect_dataset(str(tmp_path))

    assert len(result["invalid_annotations"]) == 1


def test_nonexistent_dataset() -> None:
    result = inspect_dataset("this_dataset_does_not_exist")

    assert result["image_count"] == 0
    assert result["label_count"] == 0
    assert result["errors"]
