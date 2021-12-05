"""本模块是一个小demo，用来对比单张图片结果怎么样"""

from mmdet.apis import init_detector,inference_detector, show_result_pyplot
import mmcv
import os
import glob
from PIL import Image
from torchvision.transforms import ToTensor
import torch as t

# 定义model
config_file = "/home/lry/projects/mmdetection/configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py"
checkpoint_file_before = "/home/lry/projects/mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo/latest.pth"
checkpoint_file_after = "/home/lry/checkpoints/fenghuo/model_segementation_maskrcnn_10/latest.pth"
model_before = init_detector(config_file, checkpoint_file_before,device="cuda:3")
model_after = init_detector(config_file, checkpoint_file_after,device="cuda:2")

# 选取验证集的前10张图片进行验证
imgs = glob.glob('/home/lry/data/780b_std/val/*.jpg')[:10]

for i, img in enumerate(imgs):
    result_b = inference_detector(model_before, img)
    model_before.show_result(img, result_b, out_file=f'/home/lry/checkpoints/fenghuo/compare/before/demo_val{i}.jpg')

    result_a = inference_detector(model_after, img)
    model_after.show_result(img, result_a, out_file=f'/home/lry/checkpoints/fenghuo/compare/after/demo_val{i}.jpg')


