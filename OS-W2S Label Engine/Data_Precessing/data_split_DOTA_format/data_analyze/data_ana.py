import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 后端设为Agg，无需显示图像
import matplotlib.pyplot as plt

def parse_txt(txt_path):
    """
    解析文本文件，返回 (total_images, category_dict)
    category_dict 为 {类别名称: 数量} 的字典
    """
    total_images = 0
    category_dict = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 去掉换行和空格
    lines = [line.strip() for line in lines if line.strip()]

    # 假设第一行是"图片总数: xxx"
    if lines[0].startswith("图片总数"):
        total_images = int(lines[0].split(":")[1].strip())

    # 找到 "类别统计:" 所在行
    # 然后从下一行开始都是类别和数量
    start_index = None
    for i, line in enumerate(lines):
        if "类别统计" in line:
            start_index = i + 1
            break

    if start_index is not None:
        for line in lines[start_index:]:
            if ":" in line:
                cat, num = line.split(":")
                cat = cat.strip()
                num = int(num.strip())
                category_dict[cat] = num
    return total_images, category_dict

def plot_and_save(total_images, category_dict, save_path):
    """
    根据total_images, category_dict绘制柱状图，并通过cv2保存为PNG
    """
    categories = list(category_dict.keys())
    counts = list(category_dict.values())

    # 绘制
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(categories, counts, color='skyblue', edgecolor='black')

    # 添加标题
    ax.set_title(f"Total images: {total_images}", fontsize=16)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)
    plt.xticks(rotation=45, ha='right')

    # 在柱状图上显示数值
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, str(count),
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # 将matplotlib figure转换为numpy数组
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # 保存为png
    cv2.imwrite(save_path, img)
    plt.close(fig)  # 关闭图像以释放内存

if __name__ == "__main__":
    # 假设当前目录下有多个 txt 文件
    txt_dir = "/home/disk/ICML/code/OVA-DETR-pytorch/data/RSSDD_Datasets_DOTA_Split_filter/data_analyze/dataset_level_count"
    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)

    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(txt_dir, txt_file)
            total_images, category_dict = parse_txt(txt_path)

            # 构建输出png文件名，与txt文件同名但扩展名为.png
            base_name = os.path.splitext(txt_file)[0]
            save_path = os.path.join(save_dir, base_name + ".png")

            # 绘图并保存
            plot_and_save(total_images, category_dict, save_path)
            print(f"已保存统计图至 {save_path}")
