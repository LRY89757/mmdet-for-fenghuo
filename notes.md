# MMDetection整体构建流程思想

# Preface
all reference：https://zhuanlan.zhihu.com/p/337375549
本文全篇照抄。

目前流程思想都有以下流程划分：
![整体构建流程](https://pic1.zhimg.com/80/v2-7ecc8e5e19c59a3e6682c5e3cdc34918_720w.jpg)

训练核心组件一般包括9个：
1. 特征提取，backbone层，例如ResNet层
2. 但尺度或者多尺度特征图输入到neck模块中进行特征融合或者增强，典型的neck就是FPN。
3. 一般多尺度特征最后都会输入到head部分一般包括分类和回归分支输出。
4. 整个网络构建阶段都可以引入一些即插即用增强算子来增加提取能力，例如SPP，DCN等
5. 目标检测head输出一般是特征图，对于分类任务存在严重的正负样本不平衡，可以通过正负样本属性分配和采样控制
6. 为了方便收敛和平衡多分支，一般都会对gt bbox进行编码。
7. 最后一步是计算分类和回归loss，进行训练
8. 训练过程中也包括非常多的trick,例如优化器选择，参数调节等。

但是以上9个组件不是每一个算法都需要的。


# Backbone

![](https://pic2.zhimg.com/80/v2-cdee2bd9f289d650ddbcbd748c4be0f9_720w.jpg)

backbone作用主要是特征提取，目前MMDetectiohn已经集成了大部分骨架网络，可以看一下`mmdet/models/backbones`
可以看到有：
```python
__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer', 'PyramidVisionTransformerV2'
]
```
最常用的还是ResNet系列，ResNetV1d系列和Res2Net系列。如果我们想要对骨架进行扩展，我们可以继承上述网络，然后通过注册器机制注册使用。一个典型的用法是：
```python
# 骨架的预训练权重路径
pretrained='torchvision://resnet50',
backbone=dict(
    type='ResNet', # 骨架类名，后面的参数都是该类的初始化参数
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True), 
    norm_eval=True,
    style='pytorch'),
```

通过MMCV的注册器机制，我们可以通过dict形式配置来实例化任何已经注册的类，非常方便和灵活。

# Neck
![](https://pic1.zhimg.com/80/v2-f0975c00a32fa03a80860f9c09234bbc_720w.jpg)

neck可以认为是backbone和head的连接层，主要负责对backbone的特征进行高效融合和增强，能够对输入的但尺度或者多尺度特征进行融合、增强输出等。我们可以看一下文件`mmdet/models/necks`
```python
__all__ = [
    'FPN', 'BFP', 'ChannelMapper', 'HRFPN', 'NASFPN', 'FPN_CARAFE', 'PAFPN',
    'NASFCOS_FPN', 'RFP', 'YOLOV3Neck', 'FPG', 'DilatedEncoder',
    'CTResNetNeck', 'SSDNeck', 'YOLOXPAFPN'
]

```
这个最常用的应该是FPN，一个典型的用法为：
```python
neck=dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048], # 骨架多尺度特征图输出通道
    out_channels=256, # 增强后通道输出
    num_outs=5), # 输出num_outs个多尺度特征图
```

# Head
![](https://pic2.zhimg.com/80/v2-fdd9a6232e62c75b143153dab8ba9bc1_720w.jpg)

目标检测算法输出一般包括分类和狂坐标回归两个分支，不同算法head模块复杂程度不一样，灵活度比较高。在网络构建方面，理解目标检测算法主要是要理解head模块。
MMDetection重head模块又划分为two-stage所需的ROIHEAD和one-stage所需的DenseHead,也就是说所有的one-stage算法的head模块都在`mmdet/models/dense_heads`中，而two-stage算法还包括额外的`mmdet/models/roi_heads`.

