from torchvision import models
import torch
from torch import nn, optim
import torchvision as tv
import numpy as np
from PIL import Image
import glob
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transform



# Classes = ['GFI1-1', 'O622-8', 'GFI1-2', 'XGI2', 'E1-63', 'fake', 'O2500-4', 'O9953', 'unrecognizable', 'O622-4', 'GFI2', 'others', '10GFEC', 'empty', '780B', 'GFI2-R', 'O155-8', 'GFI1-3', 'O2500']
Classes = ['GFI1-1', 'O2500622-4', 'others', 'O2500', 'O9953GFEC', 'unrecognizable', 'O622155-8', 'XGI2', 'GFI1-3', 'fake', 'GFI12R2', 'E1-63', 'empty']
num_classes = len(Classes)
print(num_classes)

# 导入数据
class data780B(Dataset):
    '''
    抽象数据类
    '''
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = glob.glob(root + '/*.jpg')
        self.Classes = ['GFI1-1', 'O2500622-4', 'others', 'O2500', 'O9953GFEC', 'unrecognizable', 'O622155-8', 'XGI2', 'GFI1-3', 'fake', 'GFI12R2', 'E1-63', 'empty']
        # self.serial = {'GFI1-1': 0, 'O622-8': 1, 'GFI1-2': 2, 'XGI2': 3, 'E1-63': 4, 'fake': 5, 'O2500-4': 6, 'O9953': 7, 'unrecognizable': 8, 'O622-4': 9, 'GFI2': 10, 'others': 11, '10GFEC': 12, 'empty': 13, '780B': 14, 'GFI2-R': 15, 'O155-8': 16, 'GFI1-3': 17, 'O2500': 18}
        self.serial = {'GFI1-1': 0, 'O2500622-4': 1, 'others': 2, 'O2500': 3, 'O9953GFEC': 4, 'unrecognizable': 5, 'O622155-8': 6, 'XGI2': 7, 'GFI1-3': 8, 'fake': 9, 'GFI12R2': 10, 'E1-63': 11, 'empty': 12}

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img) 
        # print(self.images[idx])
        label = [cls for cls in self.Classes if cls in self.images[idx]][0]
        assert type(self.serial[label]) is type(5)
        assert type(img) is type(torch.tensor([1]))
        return self.serial[label], img
        
    def __len__(self):
        return len(self.images)



transforms = transform.Compose([
    transform.Resize(180),
    transform.CenterCrop((1800, 180)),
    transform.ToTensor(),
    # transform.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

train_root = '/home/lry/data/780b_singe_padded_new/train'
val_root = '/home/lry/data/780b_singe_padded_new/val'
trainset = data780B(train_root, transform=transforms)
valset = data780B(val_root, transform=transforms)


trainloader = DataLoader(
    trainset,
    batch_size=10,
    shuffle=True,
    num_workers=16,
    drop_last=True
)

valloader = DataLoader(
    valset,
    batch_size=10,
    shuffle=True,
    num_workers =16,
    drop_last=True
)


# 导入模型
model = models.resnet50(pretrained=True)
# print(model)
print(model.fc)  # 查看默认最后一层全连接层的结构
# print(type(model.fc))
# print(dict(model.fc.named_parameters()).keys())
model.fc = nn.Linear(2048, num_classes)
print(model.fc)


# 误差
criterion = nn.CrossEntropyLoss()  # 采用交叉熵
# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.001,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)
# 训练的epoch
num_epochs = 20








def train_one_epoch(model, trainloader, valloader, optimizer, criterion):
    '''
    训练一个epoch，返回该epoch的train_loss, train_acc, val_acc准确率
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train
    correct = 0
    total = 0
    train_loss = 0.0
    for i, data in enumerate(trainloader):
        
        labels, inputs = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()

        output = model(inputs)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(output, labels)
        train_loss += loss
        loss.backward()
        optimizer.step()

    train_loss = train_loss / 2808
    train_acc = correct / total

    # val准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            labels, images = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total

    return train_loss, train_acc, val_acc

if __name__ == '__main__':
    import tqdm
    import time
    log = str(time.time())
    best_loss = 10000
    for epoch in tqdm.tqdm(list(range(num_epochs))):
        train_loss, train_acc, val_acc = train_one_epoch(model, trainloader, valloader, optimizer, criterion)
        print(f'The {epoch + 1}th epoch\'s train_loss is {train_loss}, train_acc is {train_acc}, val_acc is {val_acc}')
        best_loss = min(best_loss, train_loss)
        with open(f"{log}.log", 'a+') as f:
            f.write(f'The {epoch + 1}th epoch\'s train_loss is {train_loss}, train_acc is {train_acc}, val_acc is {val_acc}')
        
        launchTimestamp = str(time.time())
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': best_loss,
                            'optimizer': optimizer.state_dict()},
                           '/home/lry/projects/mmdetection/lry/model_classification/checkpoints' + '/resnet50-m-' + launchTimestamp + '-' + str("%.4f" % best_loss) + '.pth.tar')
        print(f'model {launchTimestamp} saved!')
