import os
import cv2
import sys
import xml.etree.ElementTree as ET
import argparse
import shutil
from tqdm import tqdm
from xml.dom import minidom
import math

def limit_value(a, b):
    if a < 1:
        a = 1
    if a >= b:
        a = b - 1
    return a

def readlabeltxt(txtpath, height, width, hbb=True):
    with open(txtpath, 'r') as f_in:
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        boxes = []
       
        for i, splitline in enumerate(splitlines):
            if len(splitline) < 10:
                raise ValueError("splitline 的长度小于 10，无法继续处理") 

            label = splitline[8]
            difficult = splitline[9]

            x1 = math.floor(float(splitline[0]))
            y1 = math.floor(float(splitline[1]))
            
            x2 = math.ceil(float(splitline[2]))
            y2 = math.floor(float(splitline[3]))
            
            x3 = math.ceil(float(splitline[4]))
            y3 = math.ceil(float(splitline[5]))
            
            x4 = math.floor(float(splitline[6]))
            y4 = math.ceil(float(splitline[7]))

            if hbb:
                xx1 = min(x1, x2, x3, x4)
                xx2 = max(x1, x2, x3, x4)
                yy1 = min(y1, y2, y3, y4)
                yy2 = max(y1, y2, y3, y4)

                xx1 = limit_value(xx1, width)
                xx2 = limit_value(xx2, width)
                yy1 = limit_value(yy1, height)
                yy2 = limit_value(yy2, height)

                # 检查框是否合理
                if xx1 < xx2 and yy1 < yy2 and \
                   xx1 >= 1 and yy1 >= 1 and \
                   xx2 <= width-1 and yy2 <= height-1:
                    box = [xx1, yy1, xx2, yy2, label, difficult]
                    boxes.append(box)
    return boxes

def write_xml(save_dir, imgname, w, h, d, bboxes):
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'labelXML'

    filename = ET.SubElement(annotation, 'filename')
    filename.text = imgname

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'RS_Sentence_Detection_Dataset'
    
    owner = ET.SubElement(annotation, 'owner')
    ow_name = ET.SubElement(owner, 'name')
    ow_name.text = 'gtwei_ann'

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(d)

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    for bbox in bboxes:
        # bbox结构：[xmin, ymin, xmax, ymax, label, difficult]
        obj = ET.SubElement(annotation, 'object')

        name = ET.SubElement(obj, 'name')
        name.text = bbox[4]

        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'

        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'

        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = str(bbox[5])

        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(bbox[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(bbox[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(bbox[2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(bbox[3])

    # 使用minidom对XML进行美化
    xml_str = ET.tostring(annotation, encoding='utf-8')
    xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="    ", encoding='utf-8')

    xmlname = os.path.splitext(imgname)[0] + '.xml'
    xml_path = os.path.join(save_dir, xmlname)
    with open(xml_path, 'wb') as f:
        f.write(xml_pretty)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process DOTA format annotations.")
    parser.add_argument("--src_root", type=str, required=True, help="Source dataset root directory.")
    # parser.add_argument("--dst_root", type=str, required=True, help="Destination directory for filtered dataset.")
    parser.add_argument("--image_exts", nargs='+', default=[".png", ".jpg", ".tif"], help="List of allowable image file extensions.")
    
    args = parser.parse_args()

    src_root = args.src_root
    image_exts = args.image_exts

    images_path = os.path.join(src_root, 'images')
    labeltxt_path = os.path.join(src_root, 'labelTxt')
    anno_new_path = os.path.join(src_root, 'labelXML')
    empty_samples_path = os.path.join(src_root, 'empty_samples')
    
    if not os.path.exists(anno_new_path):
        os.makedirs(anno_new_path)
    
    if not os.path.exists(empty_samples_path):
        os.makedirs(empty_samples_path)

    filenames = os.listdir(labeltxt_path)

    # 用于记录所有出现过的类别
    categories = set()
    difficult_num = set()
    
    for filename in tqdm(filenames):
        base = os.path.splitext(filename)[0]
        # 尝试在 images_path 下找到匹配的图像文件
        img_path = None
        for ext in image_exts:
            candidate_path = os.path.join(images_path, base + ext)
            if os.path.exists(candidate_path):
                img_path = candidate_path
                break

        if img_path is None:
            print("未找到对应的图像文件：", base, "使用的扩展名列表：", image_exts)
            continue

        im = cv2.imread(img_path)
        if im is None:
            print("无法读取图片文件：", img_path)
            continue
        (H, W, D) = im.shape

        filepath = os.path.join(labeltxt_path, filename)
        # 读取txt标签
        boxes = readlabeltxt(filepath, H, W, hbb=True)
        if len(boxes) == 0:
            print('文件为空或无有效框：', filepath)
            shutil.move(img_path, os.path.join(empty_samples_path, os.path.basename(img_path)))
            shutil.move(filepath, os.path.join(empty_samples_path, os.path.basename(filepath)))
            continue

        # 收集类别
        for b in boxes:
            categories.add(b[4])
            difficult_num.add(b[5])

        # 写入XML
        write_xml(anno_new_path, os.path.basename(img_path), W, H, D, boxes)

    # 将见过的类别写入txt文件
    categories_txt_path = os.path.join(anno_new_path, 'categories_check.txt')
    with open(categories_txt_path, 'w') as f:
        for c in sorted(categories):
            f.write(c + '\n')
        for n in sorted(difficult_num):
            f.write(n + '\n')
    # print("已将类别写入文件:", categories_txt_path)
