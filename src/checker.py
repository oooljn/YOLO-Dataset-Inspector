import os

def parse_label(label_path):
    """
    解析 YOLO 格式的标签文件，并进行格式和越界校验
    """
    if not os.path.exists(label_path):
        return None

    results = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split()
                
                if len(parts) != 5:
                    continue 
                
                try:
                    class_id, x_center, y_center, width, height = map(float, parts)
                except ValueError:
                    continue

                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    continue 
                
                results.append({
                    'class_id': int(class_id),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    except Exception:
        return []

    return results