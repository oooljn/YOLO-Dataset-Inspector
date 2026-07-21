"""
YOLO annotation visualization module.
"""

import os

from PIL import Image, ImageDraw, ImageFont


def draw_annotations(
    image_path: str,
    label_path: str,
    class_names: list[str] | None = None,
) -> Image.Image:
    """
    读取图片与 YOLO 标签，绘制标注框与类别，并返回 PIL 图片对象。

    Args:
        image_path: 图片文件路径。
        label_path: 对应的 YOLO txt 标签路径。
        class_names: 类别名称列表，下标对应 class_id。

    Returns:
        绘制完成的 PIL Image。
    """
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("simhei.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            cls_value, xc, yc, width, height = map(float, parts)
        except ValueError:
            continue

        if not cls_value.is_integer() or cls_value < 0:
            continue

        if not (0 <= xc <= 1 and 0 <= yc <= 1):
            continue

        if not (0 < width <= 1 and 0 < height <= 1):
            continue

        cls_id = int(cls_value)

        x_center = xc * img_w
        y_center = yc * img_h
        box_w = width * img_w
        box_h = height * img_h

        x1 = max(0, x_center - box_w / 2)
        y1 = max(0, y_center - box_h / 2)
        x2 = min(img_w, x_center + box_w / 2)
        y2 = min(img_h, y_center + box_h / 2)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        if class_names and 0 <= cls_id < len(class_names):
            text = class_names[cls_id]
        else:
            text = f"cls{cls_id}"

        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1), text, fill="white", font=font)

    return img


def visualize() -> None:
    """显示模块使用提示。"""
    print("Visualization module: use draw_annotations() to visualize labels.")


if __name__ == "__main__":
    visualize()
