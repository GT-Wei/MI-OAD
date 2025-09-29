import os
import shutil
from tqdm import tqdm

def filter_unseen_classes(
    dataset_path,
    delete_path,
    unseen_classes
):
    """
    过滤包含未见类别的图像和标注文件，并将它们移动到指定的删除路径。

    Args:
        dataset_path (str): 数据集的根路径，包含 images/ 和 labelTxt/ 目录。
        delete_path (str): 用于存放需要删除的文件的目标路径。
        unseen_classes (list): 需要过滤的未见类别名称列表。
    """

    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labelTxt")

    # 创建删除路径，如果不存在的话
    if not os.path.exists(delete_path):
        os.makedirs(delete_path)
        print(f"Created directory: {delete_path}")

    # 获取所有标注文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    deleted_files = 0

    for label_file in tqdm(label_files):
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 标记是否包含未见类别
        contains_unseen = False

        for line in lines:
            # 假设每行格式为: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
            parts = line.strip().split()
            if len(parts) < 9:
                continue  # 跳过格式不正确的行
            class_name = parts[8].lower()  # 转为小写以确保匹配

            if class_name in [uc.lower() for uc in unseen_classes]:
                contains_unseen = True
                break  # 一旦发现未见类别，立即跳出

        if contains_unseen:
            # 构造对应的图像文件名
            image_filename = os.path.splitext(label_file)[0] + ".jpg"  # 假设图像为jpg格式
            image_path = os.path.join(images_dir, image_filename)

            # 目标路径
            target_label_path = os.path.join(delete_path, "labelTxt")
            target_image_path = os.path.join(delete_path, "images")

            # 创建子目录，如果不存在的话
            if not os.path.exists(target_label_path):
                os.makedirs(target_label_path)
            if not os.path.exists(target_image_path):
                os.makedirs(target_image_path)

            # 移动标注文件
            shutil.copy(label_path, os.path.join(target_label_path, label_file))

            # 移动图像文件，如果存在
            if os.path.exists(image_path):
                # shutil.move(image_path, os.path.join(target_image_path, image_filename))
                shutil.copy(image_path, os.path.join(target_image_path, image_filename))
            else:
                print(f"Warning: Image file {image_filename} does not exist.")

            deleted_files += 1
            # print(f"Moved {label_file} and {image_filename} to {delete_path}")
            print(f"Copy {label_file} and {image_filename} to {delete_path}")

    print(f"\nTotal files moved: {deleted_files}")

if __name__ == "__main__":
    # 定义路径
    dataset_path = "/home/disk/ICML/datasets/ovadetr_datasets_filter/DIOR/val"
    delete_path = "/home/disk/ICML/datasets/ovadetr_datasets_filter/DIOR/zsd_eval"

    # 定义未见类别
    unseen_classes = [
        "airport",
        "basketball-court",
        "ground-track-field",
        "windmill",
        "helicopter",
        "swimming-pool",
        "bus",
        "pickup-truck",
        "truck-tractor-with-box-trailer",
        "maritime-vessel",
        "motorboat",
        "barge",
        "reach-stacker",
        "mobile-crane",
        "scraper-or-tractor",
        "excavator",
        "shipping-container-lot"
    ]

    filter_unseen_classes(dataset_path, delete_path, unseen_classes)
