"""本模块用于划分验证集和测试集，大概比例为8：2"""

import shutil
import os
import random
import glob

def move_file(imgs,  new_path):
    '''这个函数是用来移动图片从旧路径移到新路径'''

    print(new_path)
    # filelist = os.listdir(old_path) # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    filelist = imgs
    print(filelist)
    for file in filelist:
        # file = file - "/home/lry/data/780b_std/"
        src = file
        dst = os.path.join(new_path, file[len('/home/lry/data/780b_singe_padded/'):])
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)


# old_path = "/home/lry/data/780b_std"

new_val = "/home/lry/data/780b_singe_padded_new/val"
new_train = "/home/lry/data/780b_singe_padded_new/train"

pics = glob.glob("/home/lry/data/780b_singe_padded/*.jpg")
# print(pics)
print(len(pics))
# print(pics[0][-30:])
# print(len(pics))
# print(pics[0][len("/home/lry/data/780b_std/"):-5] + ".jpg")
# print(pics[0][:-4])
# print(pics)

print("moving....")
move_file(pics[:-702], new_train)
print("train end!")

print("moving...")
move_file(pics[-702:], new_val)
print("val end!")


# if __name__ == '__main__':
#     pics = glob.glob("/home/lry/data/780b_singe_padded_std/train/*.jpg")
#     print(len(pics))
