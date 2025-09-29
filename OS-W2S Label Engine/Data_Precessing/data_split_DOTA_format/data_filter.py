import os
import shutil
import argparse
from tqdm import tqdm

def main(args):
    src_root = args.src_root
    dst_root = args.dst_root
    image_exts = args.image_exts

    src_images_dir = os.path.join(src_root, "images")
    src_label_dir = os.path.join(src_root, "labelTxt")

    dst_images_dir = os.path.join(dst_root, "images")
    dst_label_dir = os.path.join(dst_root, "labelTxt")

    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    # 类别映射关系
    category_map = {
        "basketballcourt": "basketball-court",
        "groundtrackfield": "ground-track-field",
        "oiltank": "oil-tanker",
        "truck-w/box": "truck-tractor-with-box-trailer",
        "truck-w/flatbed": "truck-tractor-with-flatbed-trailer",
        "truck-w/liquid": "truck-tractor-with-liquid-tank",
        "front-loader/bulldozer": "front-loader-or-bulldozer",
        "scraper/tractor": "scraper-or-tractor",
        "hut/tent": "hut-or-tent"
    }

    label_files = [f for f in os.listdir(src_label_dir) if f.endswith('.txt')]

    # for label_file in label_files:
    for label_file in tqdm(label_files, desc="Processing label files"):
        label_path = os.path.join(src_label_dir, label_file)
        image_name = os.path.splitext(label_file)[0]
        
        # 根据image_exts判断对应的图片文件是否存在
        image_path = None
        for ext in image_exts:
            check_path = os.path.join(src_images_dir, image_name + ext)
            if os.path.exists(check_path):
                image_path = check_path
                break
        
        if image_path is None:
            # 对应图片不存在，跳过
            print('对应图片不存在，跳过')
            continue

        # 读取原标注文件
        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            # DOTA格式：x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
            parts = line.strip().split()
            if len(parts) < 9:
                # 无效行或header行，根据需要决定保留与否，这里简单跳过不合格的行。
                print("无效行或header行")
                continue

            *coords, category, difficulty = parts
            difficulty = difficulty.strip()

            # 转类别小写
            category = category.lower()

            # 类别映射
            if category in category_map:
                category = category_map[category]
                print(f"{label_path} using category_map")

            # 如果difficulty == 2，则跳过该行
            # if difficulty == '2':
            #     continue

            # 保留处理后的行
            new_line = " ".join(coords + [category, difficulty]) + "\n"
            new_lines.append(new_line)

        # 判断是否有剩余标注，如果没有则跳过该文件，并不复制对应图片
        if len(new_lines) == 0:
            print(f"{label_path} lines = 0, continute\n")
            continue

        # 将结果写入目标文件夹
        dst_label_path = os.path.join(dst_label_dir, label_file)
        with open(dst_label_path, 'w') as f:
            f.writelines(new_lines)

        # 拷贝图片
        dst_image_path = os.path.join(dst_images_dir, os.path.basename(image_path))
        if not os.path.exists(dst_image_path):
            shutil.copy2(image_path, dst_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DOTA format annotations.")
    parser.add_argument("--src_root", type=str, required=True, help="Source dataset root directory.")
    parser.add_argument("--dst_root", type=str, required=True, help="Destination directory for filtered dataset.")
    parser.add_argument("--image_exts", nargs='+', default=[".png", ".jpg", ".tif"], 
                        help="Possible image file extensions to check.")
    args = parser.parse_args()
    main(args)
