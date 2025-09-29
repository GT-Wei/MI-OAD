import os

# 标注文件目录
label_dir = "/home/disk/ICML/datasets/ovadetr_datasets/DIOR/train/labelTxt"

# 定义需要替换的类别字典
replace_dict = {
    "truck-w/box": "truck-tractor-with-box-trailer",
    "truck-w/flatbed": "truck-tractor-with-flatbed-trailer",
    "truck-w/liquid": "truck-tractor-with-liquid-tank",
    "front-loader/bulldozer": "front-loader-or-bulldozer",
    "scraper/tractor": "scraper-or-tractor",
    "hut/tent": "hut-or-tent"
}

for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)
        
        new_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # 空行直接跳过
                    print("文件为空")
                    continue
                # 按空格分割行
                parts = line.split()
                if len(parts) != 10:
                    # 若格式与预期不符，可根据需要处理，这里假设都是正确格式
                    print("格式错误")
                    new_lines.append(line)
                    continue
                
                # DOTA格式: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
                class_name = parts[8]
                difficulty = parts[9]
                
                # 如果类别需要替换，则替换之
                if class_name in replace_dict:
                    print(f'替换成功 {class_name} -> {replace_dict[class_name]}')
                    class_name = replace_dict[class_name]
                
                # 重新拼装该行
                # 顺序：x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
                new_line = " ".join(parts[0:8] + [class_name, difficulty])
                new_lines.append(new_line)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for l in new_lines:
                f.write(l + "\n")
