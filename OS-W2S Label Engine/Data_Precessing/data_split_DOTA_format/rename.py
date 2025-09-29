import os
import shutil
from tqdm import tqdm

def prefix_and_copy_files(source_dir, target_dir):
    # 确保目标目录存在，如果不存在，则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 遍历源目录下的所有文件夹和子文件夹
    for root, dirs, files in os.walk(source_dir):
        for file in tqdm(files, desc=f"正在处理 {root}", leave=False):
            # 检查文件扩展名
            if file.endswith(('.jpg', '.png', '.tif', '.txt')):
                # 定义源文件的完整路径
                source_file = os.path.join(root, file)
                # 构建目标文件路径，保留原始目录结构
                relative_path = os.path.relpath(root, source_dir)
                dataset_prefix = relative_path.split(os.sep)[0]  # 获取数据集名称作为前缀
                target_folder = os.path.join(target_dir, relative_path)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                # 为文件名添加数据集名称前缀
                new_file_name = f"{dataset_prefix}_{file}"
                # 定义目标文件的完整路径
                target_file = os.path.join(target_folder, new_file_name)
                # 复制文件
                shutil.copy2(source_file, target_file)
    print("文件已成功重命名并复制到目标目录！")

# 指定源目录和目标目录
source_dir = '/home/disk/ICML/code/OVA-DETR-pytorch/data/here'
target_dir = '/home/disk/ICML/code/OVA-DETR-pytorch/data/RSSDD_Datasets_DOTA_Split_rename'

# 调用函数
prefix_and_copy_files(source_dir, target_dir)
