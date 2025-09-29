"""
修改了DOTA官方裁剪代码
参考DescReg论文中描述的自适应裁剪算法
不手动设计Gap, 而是根据Image_size和Sub_image_size自行计算Gap, 避免Padding
"""
import os
import codecs
import numpy as np
import math
from dota_utils import GetFileFromThisRootDir
import cv2
import shapely.geometry as shgeo
import dota_utils as util
import copy
from multiprocessing import Pool
from functools import partial
import time
from tqdm import tqdm
from PIL import Image
import argparse

Image.MAX_IMAGE_PIXELS = None

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def split_single_warp(name, split_base, rate, extent):
    split_base.SplitSingle(name, rate, extent)

class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code = 'utf-8',
                #  gap=512,  # gap后续不用
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext = '.png',
                 padding=True,
                 num_process=8
                 ):
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        # self.gap = gap
        self.subsize = subsize
        # self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, 'images')
        self.labelpath = os.path.join(self.basepath, 'labelTxt')
        self.outimagepath = os.path.join(self.outpath, 'images')
        self.outlabelpath = os.path.join(self.outpath, 'labelTxt')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        self.num_process = num_process
        self.pool = Pool(num_process)
        print('padding:', padding)

        # pdb.set_trace()
        if not os.path.isdir(self.outpath):
            os.makedirs(self.outpath, exist_ok=True)
        if not os.path.isdir(self.outimagepath):
            # pdb.set_trace()
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)
        # pdb.set_trace()
    ## point: (x, y), rec: (xmin, ymin, xmax, ymax)
    # def __del__(self):
    #     self.f_sub.close()
    ## grid --> (x, y) position of grids
    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = img[up: (up + self.subsize), left: (left + self.subsize)]
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        # print(f"shape:{subimg.shape}, {subimgname}")
        h, w, c = subimg.shape
        if self.padding:
            outimg = np.zeros((self.subsize, self.subsize, c), dtype=subimg.dtype)
            outimg[0:h, 0:w, :] = subimg
            Image.fromarray(outimg).save(outdir)
        else:
            Image.fromarray(subimg).save(outdir)
            

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            #print('count:', count)
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= 0):
                    continue
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                # print('writing...')
                if (half_iou == 1):
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                elif (half_iou > 0):
                #elif (half_iou > self.thresh):
                  ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    if len(out_poly) < 4:
                        print("167 error")
                        continue

                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])

                    if (len(out_poly) == 5):
                        #print('==========================')
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue
                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)

                    for index, item in enumerate(polyInsub):
                        if (item <= 1):
                            polyInsub[index] = 1
                        elif (item >= self.subsize):
                            polyInsub[index] = self.subsize
                    outline = ' '.join(list(map(str, polyInsub)))
                    if (half_iou > self.thresh):
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        ## if the left part is too small, label as '2'
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
                #else:
                 #   mask_poly.append(inter_poly)
        self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        # img = cv2.imread(os.path.join(self.imagepath, name + extent))
        img_path = os.path.join(self.imagepath, name + extent)
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        if img is None:
            return
        img = np.array(img)
        
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = util.parse_dota_poly2(fullname)
        for obj in objects:
            obj['poly'] = list(map(lambda x:rate*x, obj['poly']))
            #obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        # outbasename = name + '__' + str(rate) + '__'
        outbasename = name + '__' # 默认rate1.0
        
        w = np.shape(resizeimg)[1]
        h = np.shape(resizeimg)[0]

        # 计算裁剪块数和重叠
        num_patches_w = w // self.subsize + (1 if w % self.subsize > 0 else 0)
        num_patches_h = h // self.subsize + (1 if h % self.subsize > 0 else 0)

        # # 为宽度和高度计算动态重叠
        # gap_w = (w % self.subsize) / (w // self.subsize) if w % self.subsize != 0 and num_patches_w > 1 else 0
        # gap_h = (h % self.subsize) / (h // self.subsize) if h % self.subsize != 0 and num_patches_h > 1 else 0
        # # Calculate the extra pixels to be distributed
        extra_w = self.subsize - w % self.subsize
        extra_h = self.subsize - h % self.subsize

        # # Adjust the subsize for each patch to distribute the extra pixels
        subsize_adjusted_w = self.subsize - extra_w // (num_patches_w) if num_patches_w > 1 else self.subsize
        subsize_adjusted_h = self.subsize - extra_h // (num_patches_h) if num_patches_h > 1 else self.subsize
        # subsize_adjusted_w = min(self.subsize + (w % self.subsize) // (num_patches_w if num_patches_w > 0 else 1), w)
        # subsize_adjusted_h = min(self.subsize + (h % self.subsize) // (num_patches_h if num_patches_h > 0 else 1), h)
        
        left, up = 0, 0
        for i in range(num_patches_w):
            if (left + self.subsize >= w):
                left = max(w - self.subsize, 0)
            up = 0
            for j in range(num_patches_h):
                if (up + self.subsize >= h):
                    up = max(h - self.subsize, 0)
                right = min(left + self.subsize, w-1)
                down = min(up + self.subsize, h-1)
                subimgname = f'{outbasename}{left}__{up}'
                self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                up += subsize_adjusted_h
            left += subsize_adjusted_w
        
    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """
        imagelist = GetFileFromThisRootDir(self.imagepath)
        
        print(f"imagelist: {len(imagelist)}")
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]

        if self.num_process == 1:
            for name in tqdm(imagenames, desc="Processing images"):
                self.SplitSingle(name, rate, self.ext)
        else:
            # For multi-process, wrap imagenames with tqdm to display progress
            # Note: tqdm needs to be used with imap or imap_unordered for better progress display with multiprocessing
            worker = partial(split_single_warp, split_base=self, rate=rate, extent=self.ext)
            list(tqdm(self.pool.imap(worker, imagenames), total=len(imagenames), desc="Processing images"))
            self.pool.close()  # 关闭进程池，阻止更多任务提交到pool
            self.pool.join()   # 等待进程池中的所有进程执行完毕
        print("split finish")

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

# if __name__ == '__main__':
#     split = splitbase(r'/home/disk/ICML/code/OVA-DETR-pytorch/data/RSSDD_Datasets_DOTA/DOTA2.0/train',
#                       r'/home/disk/ICML/code/OVA-DETR-pytorch/data/RSSDD_Datasets_DOTA_Split/DOTA2.0/train',
#                       ext='.png',  # 确保使用正确的文件扩展名
#                       gap=0,
#                       subsize=800,
#                       num_process=8,
#                       padding=True
#                     )
#     split.splitdata(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split DOTA dataset into patches.")
    parser.add_argument("--basepath", type=str, required=True, 
                        help="Base path for DOTA data (input data directory).")
    parser.add_argument("--outpath", type=str, required=True, 
                        help="Output path for split DOTA data (output directory).")
    parser.add_argument("--ext", type=str, required=True,
                        help="Image file extension, e.g., .png, .tif, etc. (default: .png)")
    parser.add_argument("--gap", type=int, default=0, 
                        help="no using!!! Gap (overlap) between patches (default: 0).")
    parser.add_argument("--subsize", type=int, default=800, 
                        help="Size of the sub-images (patch size, default: 800).")
    parser.add_argument("--num_process", type=int, default=8, 
                        help="Number of processes for multiprocessing (default: 8).")
    parser.add_argument("--padding", type=bool, default=True, 
                        help="Enable padding for smaller patches (default: True).")
    parser.add_argument("--rate", type=float, default=1.0, 
                        help="Resize rate for the images (default: 1.0).")

    args = parser.parse_args()

    split = splitbase(basepath=args.basepath,
                      outpath=args.outpath,
                      ext=args.ext,
                      gap=args.gap,
                      subsize=args.subsize,
                      num_process=args.num_process,
                      padding=args.padding)
    split.splitdata(args.rate)
