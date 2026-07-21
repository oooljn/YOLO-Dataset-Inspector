from PIL import Image, ImageDraw
import os

def visualize(img_path, label_path, out_path):
    # 打开图片
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # 读取并绘制 YOLO 格式的标签框
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, xc, yc, bw, bh = map(float, parts)
                    # 将归一化坐标转换为像素坐标
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    # 画绿色的框，线宽为2
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

    # 确保输出目录存在并保存图片
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
