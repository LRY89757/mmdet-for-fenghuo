# 本模块尝试填充非盘符区域为黑色
import cv2
import os
import numpy as np

img = cv2.imread('/home/lry/projects/mmdetection/lry/image_processing/demo/hello.jpg')

print(img.shape)


triangle = np.array([[0, 0], [2500, 1800], [2000, 3500]])


img1 = cv2.fillConvexPoly(img, triangle, (0, 0, 0))
# cv2.fillPoly(img,[contours[1]],(0,0,0))  #填充内部
cv2.imwrite('/home/lry/projects/mmdetection/lry/image_processing/demo/pad0.jpg', img)

print(img - img1)


# https://blog.csdn.net/lyxleft/article/details/90676451
