import os
import cv2
import re

def generate_yolo_labels(image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if not filename.endswith('.jpg'):
            continue
        parts = filename.split('-')
        try:
            bbox_str = parts[2]  # e.g., 256I467_388I513
            x1y1, x2y2 = bbox_str.split('_')

            # 提取数字部分
            x1, y1 = map(int, re.findall(r'\d+', x1y1))
            x2, y2 = map(int, re.findall(r'\d+', x2y2))
        except Exception as e:
            print(f"Skipping {filename} due to parsing error: {e}")
            continue

        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image could not be read: {image_path}")
            continue
        h, w = img.shape[:2]

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h

        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"Labeled: {filename}")

# 用你实际的图像路径替换掉这里
generate_yolo_labels(
    image_dir='image',
    label_dir='labels'
)
