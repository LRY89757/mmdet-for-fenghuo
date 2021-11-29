"""本模块是一个小demo，用来得出训练单张图片结果怎么样"""

from mmdet.apis import init_detector,inference_detector, show_result_pyplot
import mmcv
import os
import glob

# config_file = "/home/lry/projects/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py"
# checkpoint_file = "/home/lry/projects/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
config_file = "/home/lry/projects/mmdetection/configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py"
checkpoint_file = "/home/lry/projects/mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo/latest.pth"
model = init_detector(config_file, checkpoint_file,device="cuda:3")

# img = '/home/lry/projects/mmdetection/data/780b_std/val/IMG_20211021_153937.jpg'

# result = inference_detector(model, img)
# model.show_result(img, result, out_file='/home/lry/projects/mmdetection/lry/demo.jpg')

root = '/home/lry/data/780b_std'
imgs = glob.glob("/home/lry/data/780b_std/*.jpg")
print(imgs)
for i, img in enumerate(imgs):
    result = inference_detector(model, img)
    # model.show_result(img, result, out_file=f'/home/lry/projects/mmdetection/lry/demo{i}.jpg')
    print(result)

# imgs = "../demo/demo.jpg"
# root = "/home/lry/data/tiny_vid/dog"
# imgs = os.listdir(root)
# for img in imgs:
#     img = os.path.join(root, img)
#     result = inference_detector(model, img)
#     model.show_result(img, result, out_file=f'/home/lry/projects/mmdetection/Mydetector/dog{img[-8:]}')  # save image with result
# show_result_pyplot(model, imgs, result)

# mmcv.imwrite(result, "demo.jpg")
# print(result)
# print(type(result))

