from pathlib import Path

from PIL import Image, ImageChops

from src.visualizer import draw_annotations


def create_image(path: Path) -> Image.Image:
    image = Image.new("RGB", (200, 100), "white")
    image.save(path)
    return image


def test_draw_annotations_returns_image(tmp_path: Path) -> None:
    image_path = tmp_path / "image001.jpg"
    label_path = tmp_path / "image001.txt"

    create_image(image_path)

    label_path.write_text(
        "0 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )

    result = draw_annotations(
        str(image_path),
        str(label_path),
        ["object"],
    )

    assert isinstance(result, Image.Image)
    assert result.size == (200, 100)


def test_annotation_changes_image(tmp_path: Path) -> None:
    image_path = tmp_path / "image001.jpg"
    label_path = tmp_path / "image001.txt"

    original = create_image(image_path)

    label_path.write_text(
        "0 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )

    result = draw_annotations(
        str(image_path),
        str(label_path),
        ["object"],
    )

    difference = ImageChops.difference(
        original.convert("RGB"),
        result.convert("RGB"),
    )

    assert difference.getbbox() is not None


def test_missing_label_returns_original_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image001.jpg"
    missing_label_path = tmp_path / "missing.txt"

    original = create_image(image_path)

    result = draw_annotations(
        str(image_path),
        str(missing_label_path),
    )

    difference = ImageChops.difference(
        original.convert("RGB"),
        result.convert("RGB"),
    )

    assert difference.getbbox() is None


def test_invalid_label_does_not_crash(tmp_path: Path) -> None:
    image_path = tmp_path / "image001.jpg"
    label_path = tmp_path / "image001.txt"

    create_image(image_path)

    label_path.write_text(
        "invalid annotation\n"
        "0 1.5 0.5 0.2 0.2\n"
        "1.5 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    result = draw_annotations(
        str(image_path),
        str(label_path),
    )

    assert isinstance(result, Image.Image)
