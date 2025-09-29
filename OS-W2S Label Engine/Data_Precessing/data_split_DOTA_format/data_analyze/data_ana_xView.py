# import cv2
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端
# import matplotlib.pyplot as plt

# def plot_horizontal_bar(total_images, category_dict, save_path):
#     # 将类别按数量从大到小排序
#     sorted_categories = sorted(category_dict.items(), key=lambda x: x[1], reverse=True)
#     categories = [x[0] for x in sorted_categories]
#     counts = [x[1] for x in sorted_categories]

#     # 由于类别较多，增加图幅高度，并调整字体大小，以使类名可读
#     fig_height = len(categories)*0.3  # 根据类别数量动态调整
#     fig, ax = plt.subplots(figsize=(12, fig_height))

#     # 绘制横向条形图
#     bars = ax.barh(categories, counts, color='skyblue', edgecolor='black')

#     # 标题和坐标标签
#     ax.set_title(f"Total images: {total_images}", fontsize=14)
#     ax.set_xlabel("Count", fontsize=12)

#     # 在条形右侧显示对应的数量
#     for bar, count in zip(bars, counts):
#         width = bar.get_width()
#         ax.text(width, bar.get_y() + bar.get_height()/2, str(count),
#                 va='center', ha='left', fontsize=10)

#     plt.tight_layout()

#     # 将matplotlib figure转换为numpy数组
#     fig.canvas.draw()
#     img = np.array(fig.canvas.renderer.buffer_rgba())
#     img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

#     # 保存为png
#     cv2.imwrite(save_path, img)
#     plt.close(fig)

# if __name__ == "__main__":
#     total_images = 10446
#     category_dict = {
#         "Building": 440942,
#         "Shed": 1705,
#         "Storage-Tank": 2359,
#         "Small-Car": 272897,
#         "Cargo-Truck": 7858,
#         "Passenger-Vehicle": 3982,
#         "Shipping-Container": 1966,
#         "Shipping-container-lot": 3419,
#         "Damaged-Building": 1622,
#         "Crane-Truck": 226,
#         "Bus": 8977,
#         "Truck": 16260,
#         "Vehicle-Lot": 6549,
#         "Trailer": 5449,
#         "Truck-w/Box": 4705,
#         "Facility": 1588,
#         "Cargo-Plane": 1139,
#         "Truck-w/Flatbed": 1247,
#         "Construction-Site": 2198,
#         "Tower-crane": 237,
#         "Truck-w/Liquid": 203,
#         "Dump-Truck": 1833,
#         "Cargo-Car": 2476,
#         "Oil-Tanker": 153,
#         "Utility-Truck": 4672,
#         "Tugboat": 325,
#         "Engineering-Vehicle": 292,
#         "Front-loader/Bulldozer": 824,
#         "Mobile-Crane": 481,
#         "Excavator": 1105,
#         "Pickup-Truck": 1424,
#         "Tower": 123,
#         "Pylon": 520,
#         "Truck-Tractor": 1075,
#         "Ground-Grader": 102,
#         "Reach-Stacker": 95,
#         "Small-Aircraft": 496,
#         "Motorboat": 1925,
#         "Ferry": 320,
#         "Maritime-Vessel": 1047,
#         "Barge": 293,
#         "Scraper/Tractor": 105,
#         "Aircraft-Hangar": 393,
#         "Passenger-Car": 2163,
#         "Locomotive": 160,
#         "Container-Ship": 580,
#         "Hut/Tent": 988,
#         "Straddle-Carrier": 93,
#         "Sailboat": 859,
#         "Haul-Truck": 436,
#         "Container-Crane": 277,
#         "Fishing-Vessel": 1093,
#         "Tank-car": 150,
#         "Yacht": 644,
#         "Helicopter": 102,
#         "Helipad": 195,
#         "Cement-Mixer": 408,
#         "Fixed-wing-Aircraft": 97,
#         "Flat-Car": 183,
#         "Railway-Vehicle": 24
#     }

#     save_path = "/home/disk/ICML/code/OVA-DETR-pytorch/data/RSSDD_Datasets_DOTA_Split_rename/data_analyze/plots/horizontal_bar.png"
#     plot_horizontal_bar(total_images, category_dict, save_path)
#     print(f"Saved horizontal bar chart to {save_path}")
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import cv2

def read_xview_data(file_path):
    """Reads the xView dataset file and parses the data."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    total_images = int(lines[0].split(":")[1].strip())
    category_dict = {}
    for line in lines[2:]:  # Skip the first two lines
        category, count = line.strip().split(":")
        category_dict[category.strip()] = int(count.strip())
    return total_images, category_dict

def plot_horizontal_bar(total_images, category_dict, save_path):
    """Plots a horizontal bar chart for the given data."""
    # Sort categories by count in descending order
    sorted_categories = sorted(category_dict.items(), key=lambda x: x[1], reverse=True)
    categories = [x[0] for x in sorted_categories]
    counts = [x[1] for x in sorted_categories]

    # Adjust figure height dynamically based on number of categories
    fig_height = len(categories) * 0.3
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Create horizontal bar chart
    bars = ax.barh(categories, counts, color='skyblue', edgecolor='black')

    # Add title and labels
    ax.set_title(f"Total images: {total_images}", fontsize=14)
    ax.set_xlabel("Count", fontsize=12)

    # Annotate each bar with its value
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, str(count),
                va='center', ha='left', fontsize=10)

    plt.tight_layout()

    # Convert matplotlib figure to numpy array for saving with OpenCV
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Save as PNG
    cv2.imwrite(save_path, img)
    plt.close(fig)

# File path and output path
file_path = "/home/disk/ICML/code/OVA-DETR-pytorch/data/RSSDD_Datasets_DOTA_Split_filter/data_analyze/dataset_level_count/xView_data_analyse.txt"
save_path = "/home/disk/ICML/code/OVA-DETR-pytorch/data/RSSDD_Datasets_DOTA_Split_filter/data_analyze/dataset_level_count/xView_horizontal_bar.png"

# Read data and plot
total_images, category_dict = read_xview_data(file_path)
plot_horizontal_bar(total_images, category_dict, save_path)

