"""本模块用于批量转换labelme标记的格式，使之变成coco数据集格式"""
# -*- coding:utf-8 -*-
# !/usr/bin/env python


import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
import os


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class labelme2coco(object):

    def __init__(self, labelme_json=[], save_json_path='./tran.json'):  # 这里定义了默认的存储路径
        '''
        labelme_json: 所有labelme的json文件路径组成的列表
        save_json_path: json保存位置
        '''
        self.labelme_json = labelme_json     # labelme_json的文件路径，是一个列表存储着所有的待转化json文件路径
        self.save_json_path = save_json_path  # 保存的路径
        self.images = []  # 对应生成COCO形式json文件的image的类型。
        self.categories = []  # 同上
        self.annotations = []  # 同上
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.root = '/home/lry/projects/mmdetection/data/780b/'

        self.save_json()  # 注意这里一调用该类就直接运行相关的所有代码了，这个技巧很有用。

    def data_transfer(self):

        for num, json_file in enumerate(self.labelme_json):
            print("num:" + str(num + 1) + '    ' + json_file)
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                # 注意这个image是个函数！ 将该图片的所有相应的信息转化为COCO的image格式
                self.images.append(self.image(data, num))

                # shapes是主要的标注信息，我们重点就是读取这个，但是我们没有必要读取所有，可能只需要读取第一个780b就可以
                for shapes in data['shapes']:

                    # 首先我们会判断是否这个label是否是"780B"
                    if shapes['label'] != "780B":
                        continue

                    # 注意这里是一个循环，这个循环将所有的label有关文件都循环了一遍
                    label = shapes['label']
                    if label not in self.label:
                        # 注意这个self.categorie也是个函数，这个代码段就是要生成COCO的categories格式
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    points = shapes['points']
                    #points.append([points[0][0], points[1][1]])
                    #points.append([points[1][0], points[0][1]])
                    # 同样的，self.annotation也是一个相应的函数要将所给的points文件转化为annotation和segmentation文件
                    self.annotations.append(
                        self.annotation(points, label, num))
                    self.annID += 1


    def image(self, data, num):
        image = {}
        # 解析原图片数据，注意我们并不是所有文件都有,怪不得所有文件都打开运行会报错，真是好奇怪，但是我们可以通过文件路径打开，参见下一行代码
        try:
            img = utils.img_b64_to_arr(data['imageData'])
        except:
            img=io.imread(os.path.join(self.root, data['imagePath'])) # 通过图片路径打开图片， 不过这个就需要设一个全局的root
        # img = cv2.imread(data['imagePath'], 0)
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        # image['file_name'] = data['imagePath'].split("\\")[-1] #此行不适合我的数据集
        image['file_name'] = data['imagePath'].split("\\")[-1].split('..')[-1]

        self.height = height
        self.width = width

        return image

    def categorie(self, label):
        categorie = {}
        # categorie['supercategory'] = 'Cancer'  # 这个无用
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = label
        return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = self.getcatid(label)  # 注意，源代码默认为1
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        # print(polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        # print(index)

        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        # print(rows)
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'),
                  indent=4, cls=MyEncoder)  # indent=4 更加美观显示


# labelme_json = glob.glob(
#     '/home/lry/projects/mmdetection/data/780b/*.json')
# print(labelme_json)


# # labelme_json=['./Annotations/*.json']
# # labelme_json=glob.glob('./Annotations/*.json')

# saveme_json = '/home/lry/projects/mmdetection/lry/draft.json'
# print("Start.")

# labelme2coco(labelme_json, saveme_json)
# print("Finished.")


val_json = glob.glob('/home/lry/data/780b_std/val/*.json')
print(val_json)
saveme_json = '/home/lry/data/780b_std/val/annotation_coco.json'
print("Start.")

labelme2coco(val_json, saveme_json)
print("Finished.")

train_json = glob.glob('/home/lry/data/780b_std/train/*.json')
print(train_json)
saveme_json = '/home/lry/data/780b_std/train/annotation_coco.json'
print("Start.")

labelme2coco(train_json, saveme_json)
print("Finished.")

