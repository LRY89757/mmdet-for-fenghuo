import cv2
import os
from PIL import Image
import numpy as np
import json
import copy


def points2pad(points, img_cv):
    '''
    input:
    points:所给的四个顶点。
    img_cv:图片np.array
    output:
    需要padd的四个角的三角形的三个顶点。
    '''
    points = np.array([np.array(map(int, point)) for point in points])
    triangles = np.zeros((4, 3, 2))
    if points[0][1] < points[1][1]:
        triangles[0][0] += points[0]
        triangles[0][1] += points[3] # 左上角
    
    else:
        triangles[0][0] += points[0]
        triangles[0][1] += points[1]

    return triangles


# 从四个点得到一个大的bbox：
def points2bbox(points):
    '''
    input:
    points:所给的四个点
    img: 所给图片的信息
    output:
    bbox的左上和右下
    '''
    points = np.array(points)
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    return int(xmin), int(ymin), int(xmax), int(ymax)



root = '/home/lry/projects/mmdetection/data/780b'

with open('/home/lry/projects/mmdetection/data/780b/IMG_20211021_153937.json', 'r') as f:
    load_dict = json.load(f)

type(load_dict), load_dict.keys()

# load_dict['imagePath']

img_cv = cv2.imread(os.path.join(root, load_dict['imagePath']))

print(img_cv.shape)

shapes = load_dict['shapes']

print(len(shapes))

one_dict = shapes[3]

label = one_dict['label']
print(label)
bbox = points2bbox(one_dict['points'])
print(bbox)

# image = copy.copy(img_cv)
# draw_1=cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2],bbox[3]), (0,255,255), 8)
# cv2.imwrite("/home/lry/projects/mmdetection/lry/image_processing/demo/demo_flip.jpg", draw_1)  # 将画过矩形框的图片保存到当前文件夹

# print('flipped!\n')


for i, shape in enumerate(shapes):
    label = shape['label']
    bbox = points2bbox(shape['points'])
    print(i)
    # draw = cv2.rectangle(copy.copy(img_cv), (bbox[0], bbox[1]), (bbox[2],bbox[3]), (0,255,255), 8)
    cv2.imwrite(f'/home/lry/projects/mmdetection/lry/image_processing/demo/{i}{label}.jpg', img_cv[bbox[1]:bbox[3], bbox[0]:bbox[2]])
print('flipped!\n')




# def savfig(img):
#     '''
#     input:
#     img:图片地址
#     output:存储好的每一个小盘符的图片
#     '''
#     img_cv = cv2.imread(img)














# from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# # 首先构建detector得到我们想要的result
# checkpoints = '/home/lry/projects/mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo/latest.pth'
# config = '/home/lry/projects/mmdetection/configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py'
# img = '/home/lry/projects/mmdetection/data/780b_std/val/IMG_20211021_153937.jpg'

# model = init_detector(config=config, checkpoint=checkpoints, device='cuda:2')

# result = inference_detector(model, img)

# # show_result_pyplot(model, img, result)
# model.show_result(img, result, out_file='/home/lry/projects/mmdetection/lry/image_processing/hello.jpg')
# print(result)





