"""
YOLO annotation visualization module.
"""
from PIL import Image, ImageDraw, ImageFont
import os


def draw_annotations(
    image_path: str,
    label_path: str,
    class_names: list[str] | None = None
) -> Image.Image:
    """
    读取图片与YOLO标签，绘制标注框与类别，返回PIL图片对象
    :param image_path: 图片文件路径
    :param label_path: 对应txt标签路径
    :param class_names: 类别名称列表，下标对应class_id
    :return: 绘制完标注的PIL Image
    """
    # 打开原图并获取尺寸
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    # 字体兼容兜底
    try:
        font = ImageFont.truetype("simhei.ttf", 16)
    except:
        font = ImageFont.load_default()

    # 标签文件不存在，直接返回原图
    if not os.path.exists(label_path):
        return img

    # 读取标签文本
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 逐行解析标注
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # 过滤格式错误行（必须5个数值）
        if len(parts) != 5:
            continue
        try:
            cls_id, xc, yc, w, h = map(float, parts)
            cls_id = int(cls_id)
        except ValueError:
            continue

        # YOLO归一化坐标转像素坐标
        x_center = xc * img_w
        y_center = yc * img_h
        box_w = w * img_w
        box_h = h * img_h

        # 计算框左上角、右下角
        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2

        # 边界约束，防止框超出图片
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        # 绘制红色检测框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # 拼接类别显示文本
        if class_names and 0 <= cls_id < len(class_names):
            text = class_names[cls_id]
        else:
            text = f"cls{cls_id}"

        # 绘制带底色的类别文字
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1), text, fill="white", font=font)

    return img

def visualize():
    print("Visualization module: use draw_annotations() for full visualization function")

# 本地自测入口
if __name__ == "__main__":
    # 测试路径，根据项目目录调整
    test_img = "../examples/images/test.jpg"
    test_label = "../examples/labels/test.txt"
    test_classes = ["person", "car", "bus"]
    vis_img = draw_annotations(test_img, test_label, test_classes)
    vis_img.save("visual_result.jpg")
    vis_img.show()
