import cv2
import os
from PIL import Image
import numpy as np
import json
import copy
import glob

Classes = []

def points2pad(points, img_cv):
    '''
    input:
    points:所给的四个顶点。
    img_cv:图片np.array
    output:
    需要padd的四个角的三角形的三个顶点。
    '''
    # points = np.array([np.array(map(int, point)) for point in points])
    points = np.array([np.array(list(map(int, point))) for point in points])

    triangles = np.zeros((4, 3, 2), dtype='int32')
    xmin, ymin, xmax, ymax = points2bbox(points)
    if points[0][1] < points[1][1]:
        # 左上角
        triangles[0][0] += points[0]
        triangles[0][1] += points[3] 
        triangles[0][2] += np.array([xmin, ymin], dtype='int32')
        # 左下角
        triangles[1][0] += points[2]
        triangles[1][1] += points[3]
        triangles[1][2] += np.array([xmin, ymax], dtype='int32')
        # 右上角
        triangles[2][0] += points[0]
        triangles[2][1] += points[1]
        triangles[2][2] += np.array([xmax, ymin], dtype='int32')
        # 右下角
        triangles[3][0] += points[2]
        triangles[3][1] += points[1]
        triangles[3][2] += np.array([xmax, ymax], dtype='int32')

    else:
        # 左上角
        triangles[0][0] += points[0]
        triangles[0][1] += points[1]
        triangles[0][2] += np.array([xmin, ymin], dtype='int32')
        # 左下角
        triangles[1][0] += points[0]
        triangles[1][1] += points[3]
        triangles[1][2] += np.array([xmin, ymax], dtype='int32')
        # 右上角
        triangles[2][0] += points[2]
        triangles[2][1] += points[1]
        triangles[2][2] += np.array([xmax, ymin], dtype='int32')
        # 右下角
        triangles[3][0] += points[2]
        triangles[3][1] += points[3]
        triangles[3][2] += np.array([xmax, ymax], dtype='int32')

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

def points2padded(points, img_cv):
    triangles = points2pad(points, img_cv)
    img1 = copy.copy(img_cv)
    img1 = cv2.fillConvexPoly(img1, triangles[0], (0, 0, 0))
    img1 = cv2.fillConvexPoly(img1, triangles[1], (0, 0, 0))
    img1 = cv2.fillConvexPoly(img1, triangles[2], (0, 0, 0))
    img1 = cv2.fillConvexPoly(img1, triangles[3], (0, 0, 0))
    return img1





def segementation_single(img_json):
    with open(img_json, 'r') as f:
        load_dict = json.load(f)

    # type(load_dict), load_dict.keys()

    # load_dict['imagePath']

    img_cv = cv2.imread(os.path.join(root, load_dict['imagePath']))

    # print(img_cv.shape)

    shapes = load_dict['shapes']

    # bbox = points2bbox(one_dict['points'])

    for i, shape in enumerate(shapes):
        label = shape['label']
        if label not in Classes:
            Classes.append(label)
        points = shape['points']
        bbox = points2bbox(points)
        img1 = points2padded(points, img_cv)
        # cv2.imwrite('/home/lry/projects/mmdetection/lry/image_processing/demo/pad_demo.jpg', img1[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        try:
            cv2.imwrite(f'/home/lry/data/780b_singe_padded/{img_json[-24:-5]}{i}{label}.jpg', img1[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        except:
            pass
    print(f"image {img_json[-24:-5]} padded!\n")




if __name__ == '__main__':
    root = '/home/lry/projects/mmdetection/data/780b'
    img_jsons = glob.glob(root + '/*.json')
    # map(segementation_single, img_jsons)
    for img_json in img_jsons:
        segementation_single(img_json)
    # with open('/home/lry/data/780b_singe_padded/classes.txt', w)as f:
    #     f.write("\n".join(str(_) for _ in Classes)
    # a = ['a', 'b', 'c']
    # print(" ".join(str(_) for _ in a))  # https://www.delftstack.com/zh/howto/python/how-to-convert-a-list-to-string/

# img_jsons = glob.glob(root + '/*.json')
# print(imgs_jsons)



