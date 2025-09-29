# 将Data_pre_xml_ann(下面是很多.xml文件夹)和data_pre文件夹分割N份，方便多台机器并行处理
import os
import math
import shutil

def split_dataset(xml_folder, data_folder, output_folder, n_splits):
    """
    xml_folder:     存放 .xml 文件的路径 (例如 'Data_pre_xml_ann')
    data_folder:    存放与 .xml 文件同名文件夹的路径 (例如 'data_pre')
    output_folder:  用于输出分组数据的根目录 (例如 'split_data')
    n_splits:       要分的份数 (整数)
    """

    # 1. 获取所有 .xml 文件列表
    all_xml_files = [f for f in os.listdir(xml_folder) 
                     if f.lower().endswith('.xml')]
    all_xml_files.sort()  # 可以根据需要改成随机打乱等

    total_xml = len(all_xml_files)
    if total_xml == 0:
        print("在目录 {} 下没有找到任何 .xml 文件，脚本退出。".format(xml_folder))
        return

    # 2. 计算每一份大约多少文件
    #    如果不能整除，可以用 math.ceil 让部分分组多 1 个
    split_size = math.ceil(total_xml / n_splits)

    print("共检测到 {} 个 .xml 文件，准备分成 {} 份，每份大约 {} 个文件。"
          .format(total_xml, n_splits, split_size))

    # 如果输出目录不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 3. 按每份 split_size 大小，逐份移动
    start_idx = 0
    for i in range(n_splits):
        end_idx = start_idx + split_size
        # 切片获取当前分组包含的 .xml 文件
        current_xml_files = all_xml_files[start_idx:end_idx]
        if not current_xml_files:
            break  # 已无文件可分

        # 为本次分组创建输出目录，例如 split_1, split_2 ...
        group_dir = os.path.join(output_folder, f"split_{i+1}")
        group_xml_dir = os.path.join(group_dir, 'Data_pre_xml_ann')
        group_data_dir = os.path.join(group_dir, 'data_pre')
        os.makedirs(group_xml_dir, exist_ok=True)
        os.makedirs(group_data_dir, exist_ok=True)

        # 4. 移动当前分组的 .xml 文件 -> 对应的新目录
        for xml_file in current_xml_files:
            src_xml_path = os.path.join(xml_folder, xml_file)
            dst_xml_path = os.path.join(group_xml_dir, xml_file)

            # 移动 .xml 文件
            shutil.move(src_xml_path, dst_xml_path)

            # 找到对应的文件夹名（去掉 .xml 后缀）
            folder_name = os.path.splitext(xml_file)[0]
            src_data_path = os.path.join(data_folder, folder_name)
            dst_data_path = os.path.join(group_data_dir, folder_name)

            # 如果对应的文件夹存在，则整个移动
            if os.path.exists(src_data_path):
                shutil.move(src_data_path, dst_data_path)
            else:
                print(f"警告: {src_data_path} 不存在，可能没有对应文件夹。")

        start_idx = end_idx

    print("数据已成功分为 {} 份，输出至目录: {}".format(n_splits, output_folder))


if __name__ == "__main__":
    # 根据您的实际路径进行修改
    XML_FOLDER = "Data_pre_xml_ann"
    DATA_FOLDER = "data_pre"
    OUTPUT_FOLDER = "split_data"
    N_SPLITS = 3  # 例如要分为 3 份，可自行修改

    split_dataset(XML_FOLDER, DATA_FOLDER, OUTPUT_FOLDER, N_SPLITS)
