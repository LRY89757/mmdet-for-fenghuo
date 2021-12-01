import cv2
import os
from PIL import Image

from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# 首先构建detector得到我们想要的result
checkpoints = '/home/lry/projects/mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo/latest.pth'
config = '/home/lry/projects/mmdetection/configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py'
img = '/home/lry/projects/mmdetection/data/780b_std/val/IMG_20211021_153937.jpg'

model = init_detector(config=config, checkpoint=checkpoints, device='cuda:2')

result = inference_detector(model, img)

# show_result_pyplot(model, img, result)
model.show_result(img, result, out_file='/home/lry/projects/mmdetection/lry/image_processing/hello.jpg')
print(result)

