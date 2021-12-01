import torchvision as tv
import torch as t
import numpy as np
from PIL import Image
import glob

from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transform


class data780B(Dataset):
    '''
    抽象数据类
    '''

    Classes = [] # To Do..

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = glob.glob(root)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img) 
        label = [cls for cls in Classes if cls in self.images[idx]][0]
        return img, label
        
    def __len__(self):
        return len(self.images)



transforms = transform.Compose([
    transform.Resize(180),
    transform.CenterCrop(156, 1845),
    transform.ToTensor(),
    transform.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

root = '/home/lry/data/780b_singe_padded'
trainset = data780B(root, transform=transforms)


trainloader = DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    drop_last=False
)




        





