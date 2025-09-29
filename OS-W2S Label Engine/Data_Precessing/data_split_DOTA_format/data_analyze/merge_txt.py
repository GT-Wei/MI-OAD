def parse_analyse_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 去除换行符并清理空白
    lines = [line.strip() for line in lines if line.strip()]

    total_images = 0
    category_dict = {}
    reading_categories = False

    for line in lines:
        # 判断是否为图片总数行
        if line.startswith("图片总数:"):
            # e.g. "图片总数: 11738"
            total_images = int(line.split(":")[1].strip())
        
        # 判断是否进入类别统计段落
        elif line == "类别统计:":
            reading_categories = True
        elif reading_categories:
            # line 格式如 "bridge: 2589"
            parts = line.split(":")
            cat_name = parts[0].strip()
            cat_count = int(parts[1].strip())
            category_dict[cat_name] = cat_count

    return total_images, category_dict

def merge_results(filepaths):
    merged_total = 0
    merged_categories = {}

    for fp in filepaths:
        total, cats = parse_analyse_file(fp)
        merged_total += total
        for c, val in cats.items():
            merged_categories[c] = merged_categories.get(c, 0) + val

    return merged_total, merged_categories

# 主程序
if __name__ == "__main__":
    files = ["dataset_level_count/detail_analyse/SODA_A_test_data_analyse.txt", 
             "dataset_level_count/detail_analyse/SODA_A_val_data_analyse.txt",
             "dataset_level_count/detail_analyse/SODA_A_train_data_analyse.txt"
            ]
    merged_total, merged_categories = merge_results(files)

    # 将合并结果写入 DIOR_data_analyse.txt
    with open("dataset_level_count/SODA_A_data_analyse.txt", "w", encoding="utf-8") as f:
        f.write(f"图片总数: {merged_total}\n")
        f.write("类别统计:\n")
        for cat, cnt in merged_categories.items():
            f.write(f"{cat}: {cnt}\n")
    
    print("合并完成")
