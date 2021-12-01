"""本模块用于划分验证集和测试集，大概比例为7：3"""

import shutil
import os
import random
import glob

def move_file(old_path, imgs,  new_path):
    '''这个函数是用来移动图片和相应的json文件'''
    print(old_path)
    print(new_path)
    # filelist = os.listdir(old_path) # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    filelist = imgs
    print(filelist)
    for file in filelist:
        # file = file - "/home/lry/data/780b_std/"
        src = os.path.join(old_path, file[len("/home/lry/data/780b_std/"):])
        dst = os.path.join(new_path, file[len("/home/lry/data/780b_std/"):])
        src_json = os.path.join(old_path, file[len("/home/lry/data/780b_std/"):-5] + '.jpg')
        dst_json = os.path.join(new_path, file[len("/home/lry/data/780b_std/"):-5] + '.jpg')
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)
        shutil.move(src_json, dst_json)


old_path = "/home/lry/data/780b_std"

new_val = "/home/lry/data/780b_std/val"
new_train = "/home/lry/data/780b_std/train"

pics = glob.glob("/home/lry/data/780b_std/*.json")
# print(len(pics))
# print(pics[0][len("/home/lry/data/780b_std/"):-5] + ".jpg")
# print(pics[0][:-4])
# print(pics)

print("moving....")
move_file(old_path, pics[:140], new_train)
print("train end!")

print("moving...")
move_file(old_path, pics[140:], new_val)
print("val end!")
