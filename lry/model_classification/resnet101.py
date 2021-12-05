import torchvision as tv
import torch as t
import numpy as np
from PIL import Image
import glob

from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as transform
from torchvision import models


# 导入数据
class data780B(Dataset):
    '''
    抽象数据类
    '''
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = glob.glob(root + '/*.jpg')
        self.Classes = ['GFI1-1', 'O622-8', 'GFI1-2', 'XGI2', 'E1-63', 'fake', 'O2500-4', 'O9953', 'unrecognizable', 'O622-4', 'GFI2', 'others', '10GFEC', 'empty', '780B', 'GFI2-R', 'O155-8', 'GFI1-3', 'O2500']
        self.serial = {'GFI1-1': 0, 'O622-8': 1, 'GFI1-2': 2, 'XGI2': 3, 'E1-63': 4, 'fake': 5, 'O2500-4': 6, 'O9953': 7, 'unrecognizable': 8, 'O622-4': 9, 'GFI2': 10, 'others': 11, '10GFEC': 12, 'empty': 13, '780B': 14, 'GFI2-R': 15, 'O155-8': 16, 'GFI1-3': 17, 'O2500': 18}

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img) 
        label = [cls for cls in self.Classes if cls in self.images[idx]][0]
        return self.serial[label], img
        
    def __len__(self):
        return len(self.images)



transforms = transform.Compose([
    transform.Resize(180),
    transform.CenterCrop((1800, 180)),
    transform.ToTensor(),
    # transform.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

train_root = '/home/lry/data/780b_singe_padded_std/train'
val_root = '/home/lry/data/780b_singe_padded_std/val'
trainset = data780B(train_root, transform=transforms)
valset = data780B(val_root)


trainloader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=16,
    drop_last=True
)

valloader = DataLoader(
    valset,
    batch_size=128,
    shuffle=True,
    num_workers =16,
    drop_last=True
)

# print(trainloader[0])
for data in trainloader:
    labels, imgs = data[0], data[1]
    # print(labels, type(imgs))
    labels = list(labels)
    labels = t.tensor(list(labels))
    print(labels)
    break

# tup = (1, 2, 3, 4)
# tup_ = t.tensor(tup)
# print(tup_)

# print(trainset[8], len(trainset))


# trainiter = iter(trainloader)
# print(trainiter.next())
# valiter = iter(valloader)
# print(valiter.next())

# 保存其中几张图片看一下效果
# labels, imgs = next(trainiter)
# print(labels, imgs[0])
# img = transform.ToPILImage()(imgs[0])
# img.save('/home/lry/projects/mmdetection/lry/image_processing/demo/demo_pro.jpg')


# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/df1f5ef1c1a8e1a111e88281b27829fe/finetuning_torchvision_models_tutorial.ipynb



# 定义模型

# model = models.resnet50(pretrained=False)
# print(model)








