import os
import cv2
import numpy as np
from PIL import Image

def parse_dota_poly(filename):
    """
    从DOTA格式的txt标注文件中解析出多边形标注结果。
    格式一般为（每一行对应一个目标）:
    x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
    """
    objects = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('imagesource') or line.startswith('gsd'):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue

            poly = [
                (float(parts[0]), float(parts[1])),
                (float(parts[2]), float(parts[3])),
                (float(parts[4]), float(parts[5])),
                (float(parts[6]), float(parts[7]))
            ]
            category = parts[8]
            difficult = parts[9] if len(parts) > 9 else '0'
            obj = {
                'poly': poly,
                'name': category,
                'difficult': difficult
            }
            objects.append(obj)
    return objects

def load_image(image_path):
    # 使用Pillow读取各种格式的图片，包括tif、png、jpg
    img_pil = Image.open(image_path)
    # 将PIL image转换为NumPy数组（RGB格式）
    img = np.array(img_pil)
    # 如果需要使用OpenCV进行处理，通常OpenCV使用BGR格式:
    # 因为PIL读取的是RGB，为统一，可转换为BGR
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif len(img.shape) == 2:
        # 灰度图则可直接使用，也可转为BGR以统一处理
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def visualize_dota_image(image_path, txt_path, output_path, show_label=True):
    # 读取图像（支持jpg/tif）
    img = load_image(image_path)
    # 解析TXT标注
    objects = parse_dota_poly(txt_path)

    # 绘制多边形标注
    for obj in objects:
        poly = obj['poly']
        category = obj['name']
        # 将poly转成numpy格式 (N,1,2)
        poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

        # 随机颜色
        color = (np.random.randint(0, 255), 
                 np.random.randint(0, 255), 
                 np.random.randint(0, 255))

        # 绘制多边形
        cv2.polylines(img, [poly_np], isClosed=True, color=color, thickness=3)

        # 在多边形的第一个顶点位置写上类别名称（可选）
        if show_label:
            x, y = poly[0]
            cv2.putText(img, category, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, color, 2, cv2.LINE_AA)

    # 保存结果图像
    cv2.imwrite(output_path, img)
    print("Visualization saved to:", output_path)

if __name__ == '__main__':
    # 示例用法
    # 可以使用任意.jpg或.tif图像以及对应的DOTA格式标注txt
    image_path = '/home/disk/ICML/datasets/RSSDD_Datasets_DOTA_Split_filter/DIOR/val/images/DIOR_09455.jpg'  # or '/path/to/your_image.jpg'
    txt_path = '/home/disk/ICML/datasets/RSSDD_Datasets_DOTA_Split_filter/DIOR/val/labelTxt/DIOR_09455.txt'
    output_path = '/home/disk/ICML/datasets/data_split_DOTA_format/test_output/vis_image.png'
    visualize_dota_image(image_path, txt_path, output_path, show_label=True)
