import os
import argparse
from collections import defaultdict

def count_images_and_labels(image_dir, label_dir):
    # 初始化类别统计字典
    label_count = defaultdict(int)

    # 获取所有标注文件
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    # 遍历所有标注文件
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
            # 检查是否有有效的标注
            valid_labels = [line for line in lines if len(line.split()) >= 10]

            if not valid_labels and lines:
                print('注释有误')
            
            if not lines:
                print(f"remove {label_path}")
                # 如果标注为空，则删除标注文件和对应的图片文件
                os.remove(label_path)
                # 图片文件可能具有不同的扩展名，尝试删除常见格式
                image_extensions = ['.jpg', '.png', '.tif']
                for ext in image_extensions:
                    image_path = os.path.join(image_dir, label_file.replace('.txt', ext))
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        break
            else:
                # 如果标注有效，更新类别计数
                for line in valid_labels:
                    label = line.split()[-2]
                    label_count[label] += 1

    # 统计剩余的图片数量
    image_count = len([name for name in os.listdir(image_dir) if name.endswith(('.jpg', '.png', '.tif'))])
    
    return image_count, dict(label_count)

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="统计图片数量和标注类别信息，并删除空标注及对应图片")
    parser.add_argument('dataset_path', type=str, help="数据集的基本路径")
    args = parser.parse_args()
    
    # 构建图片和标注的目录路径
    image_dir = os.path.join(args.dataset_path, 'images')
    label_dir = os.path.join(args.dataset_path, 'labelTxt')
    
    # 调用统计函数
    image_count, label_stats = count_images_and_labels(image_dir, label_dir)
    
    
     # 提取数据集目录名称并拼接文件名
    dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
    result_file_name = f"{dataset_name}_data_analyse.txt"
    result_file_path = os.path.join(args.dataset_path, result_file_name)

    with open(result_file_path, 'w') as result_file:
        result_file.write(f"图片总数: {image_count}\n")
        result_file.write("类别统计:\n")
        for label, count in label_stats.items():
            result_file.write(f"{label}: {count}\n")

    print(f"分析结果已保存到 {result_file_path}")

if __name__ == '__main__':
    main()
