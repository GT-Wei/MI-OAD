import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_dataset_file(file_path):
    """
    Parse a single dataset statistics file, returns total images and a dict of category counts.
    Expected file format:
    图片总数: 23463
    类别统计:
    bridge: 3956
    harbor: 5469
    ...
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines if l.strip()]

    total_images = 0
    category_stats = {}
    mode = None

    for line in lines:
        if line.startswith("图片总数"):
            # Format: "图片总数: 23463"
            total_images = int(line.split(":")[1].strip())
        elif line.startswith("类别统计"):
            mode = "category"
        elif mode == "category":
            # Format: bridge: 3956
            if ":" in line:
                cat, count = line.split(":")
                cat = cat.strip().lower()  # Convert category to lowercase
                count = int(count.strip())
                category_stats[cat] = count

    return total_images, category_stats

def main(directory_path, output_txt, output_fig):
    # Read all txt files from the given directory
    files = [f for f in os.listdir(directory_path) if f.lower().endswith('.txt')]

    if len(files) != 8:
        print(f"The directory {directory_path} does not contain 8 txt files. Found: {len(files)}")
        sys.exit(1)

    total_images_sum = 0
    category_dict = defaultdict(lambda: {"total": 0, "datasets": {}})

    for f in files:
        file_path = os.path.join(directory_path, f)
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        images, stats = parse_dataset_file(file_path)
        total_images_sum += images
        for cat, count in stats.items():
            category_dict[cat]["total"] += count
            category_dict[cat]["datasets"][dataset_name] = count

    # Sort categories in alphabetical order (all are lowercase now)
    sorted_categories = sorted(category_dict.keys())

    # Total instances from all datasets
    all_categories_sum = sum(cat_info["total"] for cat_info in category_dict.values())

    # Write results to output_txt in English
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"Total number of images from all datasets: {total_images_sum}\n")
        f.write(f"Total number of instances from all datasets: {all_categories_sum}\n")
        f.write("Category details (alphabetical order):\n")

        for i, cat in enumerate(sorted_categories, start=1):
            cat_total = category_dict[cat]["total"]
            detail_str = " + ".join([f"{ds}:{count}" for ds, count in category_dict[cat]["datasets"].items()])
            f.write(f"{i}. {cat}: {cat_total} ({detail_str})\n")

    # Draw a horizontal bar chart
    counts = [category_dict[cat]["total"] for cat in sorted_categories]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_categories, counts, color='skyblue')
    plt.xlabel("Number of Instances")
    plt.ylabel("Categories")
    plt.title("Category-Instance Count Statistics")

    # Add count labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.0,
                 f"{width}", ha='left', va='center', fontsize=8)

    # Add total images and total instances info as a subtitle
    plt.text(0.5, 1.02, f"Total images: {total_images_sum} | Total instances: {all_categories_sum}",
             ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <directory_path> <output_txt> <output_fig>")
        sys.exit(1)

    dir_path = sys.argv[1]
    output_txt = sys.argv[2]
    output_fig = sys.argv[3]
    main(dir_path, output_txt, output_fig)
