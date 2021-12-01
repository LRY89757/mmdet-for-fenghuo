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




---
title: mmdet && project of fenghuo
author: 逯润雨
top: false
cover: false
toc: true
mathjax: false
date: 2021-11-09 19:22:50
img:
coverImg:
keywords:
password:
summary: 
  本博客记录烽火项目本人的做项目的技术过程。
tags:
 - 深度学习
 - 项目
 - 烽火集团
categories:
 - 深度学习
---





# Preface

> 2021.11.9日更新：

本博客记录烽火项目本人的做项目的技术过程。首先------我不会分割呢还😣😣😣，我就大概知道分割是什么另外就是数据集大致形式……（~~虽然目前来看也够了……~~）

这就是"干中学"嘛😭😭。

幸亏的是，之前看的许多官方文档我都记录了相关的过程，不需要重复看文档了，而且。一个更好的消息就是我当时看的官方文档几乎都是关于MaskRCNN的，我实际上对这些东西的测试训练的很多流程非常清楚了（~~只是一部分，一小部分~~），这就为我虽然不熟悉、不太会MaskRCNN但是不会对我训练、对我推理相关模型产生一些很大的障碍，我或许只需要就想在做检测一样来做这一件事就好了。当然输入和输出还是要我自己来搞定的，这也是最不好弄的一部分。

由于本月月末会有两场比较重要的考试，所以最好我能在本周大致写出一部分代码然后debug一些东西，这两周可以适当多参加一下项目然后到本月后一周开始准备考试。

# 2021.11.9日晚更新

差不多确定计划之后，这一晚上还是先看看我之前的所有关于MMDetection的博客，对于训练和测试以及数据集重新熟悉整合一下。

**Note**: MMDetection only supports evaluating mask AP of dataset in COCO format for now. So for instance segmentation task users should convert the data into coco format.

## Config
### dataset prepare

这里我初步打算根据我们的json文件写一个转化为COCO数据集的`.py`文件，因为组长推荐的写一个dataloader这种方式我目前没有太明白，反正不管黑猫白猫,先写出来能跑就行😞。


COCO数据集大概长这个样子：
```python
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
```
当时看官网后写的博客当时第一步就是转化一下数据集，当时官网专门提供了一个Ballon数据集，也是二分类的数据集，当时数据集的格式确实不是非常合理看着，后阿里经过转化之后变成了COCO的标准格式，我们这里重新来看一下,吸取上次的教训，这里选择分层来看结构：
* 第一层结构： 
[![](https://s6.jpg.cm/2021/11/09/I967LC.png)](https://imagelol.com/image/I967LC)
大结构就是图像都有什么，然后标记都是什么，然后是分类的类别都有什么.接下来深入看：
* 第二层结构之`images`
[![](https://s6.jpg.cm/2021/11/09/I96d4R.png)](https://imagelol.com/image/I96d4R)

`images`主要记录的就是图片的一个ids还有高宽，这个高宽还是蛮重要的，以及文件名。

* 第二层结构之`annotations`
[![](https://s6.jpg.cm/2021/11/09/I96eDz.png)](https://imagelol.com/image/I96eDz)
该层就是我们的分类category_id和框bbox还有分割数据segmentation的结构了。category_id是类别的种类数，bbox是检测框，segmentation是分割的数据标注。其他的是和上面的一一对应的（~~比如说image_id表示该标注属于那一张图片……~~），还有一些不说了，不是非常重要。
看下完整的包含具体bbox和segmentation数据的图：
[![](https://s6.jpg.cm/2021/11/09/I96U2p.png)](https://imagelol.com/image/I96U2p)

* 第二层结构之`categories`:
[![](https://s6.jpg.cm/2021/11/09/I96Y9W.png)](https://imagelol.com/image/I96Y9W)

由于这个数据集非常简单,就是一个关于balloon的检测分割任务,所以类别只有一个,相对而言还挺好.
~~我突然蜜汁自信,我们的数据也是初步只有一个框让我们检测,我怎么觉得我又行了~~

好滴,数据集大概就是这样!我们终于可以来看一下关于Config相关的文件了.

### config

我自认为我当时写得关于Config的解释已经够详细了直到我又看了一遍相关的[官方文档](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html),发现当时解释的漏洞还是非常多,所以在这里多写一点.

> We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments. If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

正如上面我们可以通过`python tools/misc/print_config.py /PATH/TO/CONFIG` 命令来对我们想要了解的config文件来看他们相关的具体配置.~~亏我当时还专门写了个相关的打印配置代码...~~

抄一下我自己的blog:
> 我们的config需要放到`configs/balloon/`目录下并且命名为：`mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py`
>
> 关于这么一个名字，为什么这么复杂是有原因的，参见这里:https://mmdetection.readthedocs.io/zh_CN/v2.18.0/tutorials/config.html#id4

当时的我也真是够辛苦,官网给的挺多东西都有问题,我当时就自己修改了许多东西,总算让我们的balloon.py成功运行了:
```python
# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('balloon',)
data = dict(
    train=dict(
        img_prefix='/home/lry/projects/mmdetection/data/balloon/train/',
        classes=classes,
        ann_file='/home/lry/projects/mmdetection/data/balloon/train/annotation_coco.json'),
    val=dict(
        img_prefix='/home/lry/projects/mmdetection/data/balloon/val/',
        classes=classes,
        ann_file='/home/lry/projects/mmdetection/data/balloon/val/annotation_coco.json'),
    test=dict(
        img_prefix='/home/lry/projects/mmdetection/data/balloon/val/',
        classes=classes,
        ann_file='/home/lry/projects/mmdetection/data/balloon/val/annotation_coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
```

当时写那篇博文[practice mmdet(Customized datasets)](https://lry89757.github.io/2021/10/12/practice-mmdet-customized-datasets/)的时候,还没有彻底读源码进行相关的溯源,而在我溯源代码[Code Trace of mmdetection](https://lry89757.github.io/2021/10/16/code-trace-of-mmdetection/)之后,对整个工程的理解上升了一个层次,不得不说还真是够可以的,还是学好相关的文件、读懂源码才算真正理解了很多东西. 而尽管读了部分源码但还是不知全局,正是当我看了知乎官方写得整体框架后才更加全局了解了有关的结构.对于CV整体的流程目前也有了更好的把握.

上面一段扯了那么多,我们来运行命令`python tools/misc/print_config.py /home/lry/projects/mmdetection/configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py  `来看看config到底是什么东西:
我先把得到的整个的结果复制下来以便各位研究:
```python
Config:
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
dataset_type = 'COCODataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/home/lry/projects/mmdetection/data/balloon/train/annotation_coco.json',
        img_prefix='/home/lry/projects/mmdetection/data/balloon/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='Resize',
                img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                           (1333, 768), (1333, 800)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        classes=('balloon', )),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/lry/projects/mmdetection/data/balloon/val/annotation_coco.json',
        img_prefix='/home/lry/projects/mmdetection/data/balloon/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('balloon', )),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/lry/projects/mmdetection/data/balloon/val/annotation_coco.json',
        img_prefix='/home/lry/projects/mmdetection/data/balloon/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('balloon', )))
evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
resume_from = None
workflow = [('train', 1)]
classes = ('balloon', )
```

我们一个一个慢慢看,实际上就是检测分割的各个结构流程(backbone, neck, rpn_head, roi_head)以及我们训练验证数据的格式预训练方法(train_cfg, data_type, data_root, train_pipeline, test...)以及各类关于权重位置、学习率、优化函数等等...:

![](https://gitee.com/moisten-the-rain/image01/raw/master/img/20211109210515.png)

![](https://gitee.com/moisten-the-rain/image01/raw/master/img/20211109210526.png)

![](https://gitee.com/moisten-the-rain/image01/raw/master/img/20211109210532.png)

![](https://gitee.com/moisten-the-rain/image01/raw/master/img/20211109210556.png)

![](https://gitee.com/moisten-the-rain/image01/raw/master/img/20211109210539.png)

![](https://gitee.com/moisten-the-rain/image01/raw/master/img/20211109210544.png)

除了以上结构之外,剩下的就是一些我们的其余配置:
```python
evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
resume_from = None
workflow = [('train', 1)]
classes = ('balloon', )
```



# 2021.11.10晚更新

## Preface 

按照昨天看的自己之前博客和总结的思路，目前当务之急就是找到我们需要的代码来转换labelme标注的数据集变为COCO格式的，目前在GitHub我找到了一些相关的开源小项目，打算根据这个仓库https://github.com/veraposeidon/labelme2Datasets 来转换我们的数据，当然这里肯定需要更改一些代码。另外这里由于分割并没有做数据标注，所以我自己得重新写一遍分割的相关标注。



> 真是懒得够可以，11.10整完的，要等到11.13号来写，吐了。

## COCO.json和Labelme.json区别
~~待更~~
2021.11.13已更新。

上面有提到Labelme.json文件主要格式，COCO数据集大概长这个样子：
```python
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
```
COCO的json文件是所有图片都放到一个json文件中，并且类别、图片提前都用编号id定义好。



我们可以继续看一下Labelme.json文件包含什么：

```python
{
  "version": "4.5.13",
  "flags": {},
  "shapes": [
    {
      "label": "780B",
      "points": [
        [
          734.0,
          2425.0
        ],
        [
          2779.0,
          2355.0
        ],
        [
          2584.0,
          3580.0
        ],
        [
          819.0,
          3770.0
        ]
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    },
    ...
    {
    "label":"780B",
    "points":[
    	...
    ],
    "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "IMG_20211021_154043.jpg",
  "imageData": null,
  "imageHeight": 4608,
  "imageWidth": 3456
}
```
labelme文件是每一个图片都有自己的json文件负责存储各个图片。

另外这里介绍一下COCO数据集中的segmentation类型的label是什么样子的，实际上就是类似于多边形的每一个顶点这个样式，这对我们目前的数据集非常方便，因为我们目前数据集并没有标注每一个物体的分割数据集，但是我们的目标本身就是矩形框，所以我们直接用检测的label就可以直接转化为COCO格式的分割的label。因为都是只有4个点就可以。





## 转换源码初始版 


目前打算转换的源码如下：
```python
"""本模块用于批量转换labelme标记的格式，使之变成coco数据集格式"""
# -*- coding:utf-8 -*-
# !/usr/bin/env python


import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image


class MyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(MyEncoder, self).default(obj)


class labelme2coco(object):
	def __init__(self, labelme_json=[], save_json_path='./tran.json'):
		'''
		labelme_json: 所有labelme的json文件路径组成的列表
		save_json_path: json保存位置
		'''
		self.labelme_json = labelme_json
		self.save_json_path = save_json_path
		self.images = []
		self.categories = []
		self.annotations = []
		# self.data_coco = {}
		self.label = []
		self.annID = 1
		self.height = 0
		self.width = 0

		self.save_json()

	def data_transfer(self):

		for num, json_file in enumerate(self.labelme_json):
			print("num:" + str(num + 1) + '    ' + json_file)
			with open(json_file, 'r') as fp:
				data = json.load(fp)  # 加载json文件
				self.images.append(self.image(data, num))
				for shapes in data['shapes']:
					label = shapes['label']
					if label not in self.label:
						self.categories.append(self.categorie(label))
						self.label.append(label)
					points = shapes['points']  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
					#points.append([points[0][0], points[1][1]])
					#points.append([points[1][0], points[0][1]])
					self.annotations.append(self.annotation(points, label, num))
					self.annID += 1

	def image(self, data, num):
		image = {}
		img = utils.img_b64_to_arr(data['imageData'])  # 解析原图片数据
		# img=io.imread(data['imagePath']) # 通过图片路径打开图片
		# img = cv2.imread(data['imagePath'], 0)
		height, width = img.shape[:2]
		img = None
		image['height'] = height
		image['width'] = width
		image['id'] = num + 1
		#image['file_name'] = data['imagePath'].split("\\")[-1] #此行不适合我的数据集
		image['file_name'] = data['imagePath'].split("\\")[-1].split('..')[-1]

		self.height = height
		self.width = width

		return image

	def categorie(self, label):
		categorie = {}
		categorie['supercategory'] = 'Cancer'
		categorie['id'] = len(self.label) + 1  # 0 默认为背景
		categorie['name'] = label
		return categorie

	def annotation(self, points, label, num):
		annotation = {}
		annotation['segmentation'] = [list(np.asarray(points).flatten())]
		annotation['iscrowd'] = 0
		annotation['image_id'] = num + 1
		# annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
		# list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
		annotation['bbox'] = list(map(float, self.getbbox(points)))
		annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
		# annotation['category_id'] = self.getcatid(label)
		annotation['category_id'] = self.getcatid(label)  # 注意，源代码默认为1
		annotation['id'] = self.annID
		return annotation

	def getcatid(self, label):
		for categorie in self.categories:
			if label == categorie['name']:
				return categorie['id']
		return 1

	def getbbox(self, points):
		# img = np.zeros([self.height,self.width],np.uint8)
		# cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
		# cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
		polygons = points

		mask = self.polygons_to_mask([self.height, self.width], polygons)
		#print(polygons)
		return self.mask2box(mask)

	def mask2box(self, mask):
		'''从mask反算出其边框
		mask：[h,w]  0、1组成的图片
		1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
		'''
		# np.where(mask==1)
		index = np.argwhere(mask == 1)
		#print(index)


		rows = index[:, 0]
		clos = index[:, 1]
		# 解析左上角行列号
		#print(rows)
		left_top_r = np.min(rows)  # y
		left_top_c = np.min(clos)  # x

		# 解析右下角行列号
		right_bottom_r = np.max(rows)
		right_bottom_c = np.max(clos)

		# return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
		# return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
		# return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
		return [left_top_c, left_top_r, right_bottom_c - left_top_c,
		        right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

	def polygons_to_mask(self, img_shape, polygons):
		mask = np.zeros(img_shape, dtype=np.uint8)
		mask = PIL.Image.fromarray(mask)
		xy = list(map(tuple, polygons))
		PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
		mask = np.array(mask, dtype=bool)
		return mask

	def data2coco(self):
		data_coco = {}
		data_coco['images'] = self.images
		data_coco['categories'] = self.categories
		data_coco['annotations'] = self.annotations
		return data_coco

	def save_json(self):
		self.data_transfer()
		self.data_coco = self.data2coco()
		# 保存json文件
		json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示


labelme_json = glob.glob('H:/camellia/Aug/训练/val_json/*.json')
print(labelme_json)

# labelme_json=['./Annotations/*.json']
# labelme_json=glob.glob('./Annotations/*.json')

saveme_json = 'H:/camellia/Aug/训练/annotations/val.json'
print("Start.")

labelme2coco(labelme_json, saveme_json)
print("Finished.")
```


## 代码详解
按照运行顺序来解释代码：

### main()
关于以上代码，我们可以根据运行的顺序来定义，首先是glob了所有.json前缀的文件，因为我们labelme的文件都是一个图片专门生成一个.json文件，而我们的COCO数据集格式是所有的图片的信息都放到了一个.json图片中，接着我们定义保存到的json文件，然后直接调用类labelme2coco,然后直接得到的结果。


### labelme2coco
labelme2coco是整个文件的关键。

### `__init__`
```python
class labelme2coco(object):
	def __init__(self, labelme_json=[], save_json_path='./tran.json'):  # 这里定义了默认的存储路径
		'''
		labelme_json: 所有labelme的json文件路径组成的列表
		save_json_path: json保存位置
		'''
		self.labelme_json = labelme_json     # labelme_json的文件路径，是一个列表存储着所有的待转化json文件路径
		self.save_json_path = save_json_path  # 保存的路径
		self.images = []  # 对应生成COCO形式json文件的image的类型。
		self.categories = [] # 同上
		self.annotations = [] # 同上
		# self.data_coco = {}
		self.label = []
		self.annID = 1
		self.height = 0
		self.width = 0

		self.save_json()  # 注意这里一调用该类就直接运行相关的所有代码了，这个技巧很有用。
```
我们接下来应该看`save_json()`：


### save_json()

```python
	def save_json(self):
		self.data_transfer()
		self.data_coco = self.data2coco()
		# 保存json文件
		json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示
```
这个可以看到实际上就是一个调用的类内的函数来总体调用。我们继续来看data_transfer()

### data_transfer()
```python
	def data_transfer(self):

		for num, json_file in enumerate(self.labelme_json):
			print("num:" + str(num + 1) + '    ' + json_file)
			with open(json_file, 'r') as fp:
				data = json.load(fp)  # 加载json文件
				self.images.append(self.image(data, num))  # 注意这个image是个函数！ 将该图片的所有相应的信息转化为COCO的image格式
				for shapes in data['shapes']:   # shapes是主要的标注信息，我们重点就是读取这个，但是我们没有必要读取所有，可能只需要读取第一个780b就可以
					label = shapes['label']  # 注意这里是一个循环，这个循环将所有的label有关文件都循环了一遍
					if label not in self.label:
						self.categories.append(self.categorie(label))  # 注意这个self.categorie也是个函数，这个代码段就是要生成COCO的categories格式
						self.label.append(label)
					points = shapes['points']  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
					#points.append([points[0][0], points[1][1]])
					#points.append([points[1][0], points[0][1]])
					self.annotations.append(self.annotation(points, label, num)) # 同样的，self.annotation也是一个相应的函数要将所给的points文件转化为annotation和segmentation文件
					self.annID += 1
```

这里一步一步说，首先就是要将所有的json文件循环遍历一遍，之后打开每个文件，首先加载文件，之后调用类中定义好的self.image()函数,这个函数将该图片所有相应的信息转化成COCO的image的格式然后返回COCO格式的信息，我们将这个信息加到self.images中去。我们先解析一下相关的self.image()函数。

### image()
实际上image()需要的信息也就是一个图片的高宽、将图片整理成编号，图片名。
```python
image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}
```
image函数实际上就是读取了以上的提供的信息然后转化过去：
```python
	def image(self, data, num):
		image = {}
		img = utils.img_b64_to_arr(data['imageData'])  # 解析原图片数据，注意我们并不是所有文件都有,怪不得所有文件都打开运行会报错，真是好奇怪，但是我们可以通过文件路径打开，参见下一行代码
		# img=io.imread(os.path.join(root, data['imagePath'])) # 通过图片路径打开图片， 不过这个就需要设一个全局的root
		# img = cv2.imread(data['imagePath'], 0)
		height, width = img.shape[:2]
		img = None
		image['height'] = height
		image['width'] = width
		image['id'] = num + 1
		#image['file_name'] = data['imagePath'].split("\\")[-1] #此行不适合我的数据集
		image['file_name'] = data['imagePath'].split("\\")[-1].split('..')[-1]

		self.height = height
		self.width = width

		return image
```
注意到我们最终返回的是一个字典。
这里我们发现用到了一个函数utils.img_b64_to_arr()，这个函数是用来将我们的图片读取成cv2的img格式矩阵，这个utils实际上是labelme这个库中的一个模块。目前我的理解是，labelme生成的.json文件里面的imageData是一堆我看不懂的编码，这些编码实际上存储的就是图片的信息，然后我们通过`img_b64_to_arr()`函数来“翻译”为我们能看懂的图片RGB矩阵形式存储。
可以看到下面的注释也给出了一些其他方式的打开图片形式，有用cv2读取，有用skimage.io读取的，因为有可能我们data['imageData']的值为null(~~我目前项目的数据集就是这样~~)。其他的格式都是很简单的复制了，相对而言不用多说。

### data_transfer()

好了我们接下来继续说我们的data_transfer()函数。接下来是对data['shapes']所有文件进行一个遍历，shapes是主要的标注信息，我们重点就是读取这个，shapes里面包含着所有的标注框ground truth信息，当然我们这里因为仅仅对一个大框做分割所以只需要一个780B的label就可以，这里它将所有都遍历了一遍，然后调用了self.categorie函数来让我们的label里面的信息转化为COCO格式的。实际上就是变为COCO格式json文件中那个categories类型，我们继续来看self.categorie()

### categorie()
```python
	def categorie(self, label):
		categorie = {}
		categorie['supercategory'] = 'Cancer'
		categorie['id'] = len(self.label) + 1  # 0 默认为背景
		categorie['name'] = label
		return categorie
```
实际上非常简单，因为COCO里面的categories只需要知道整个数据集有哪些类，每个类都有一个对应的id就OK了。

### data_transfer()
实际上后面都是类似的，对应的功能调用相应的函数就OK。这里不详细谈了，没有必要详细解释。

### data2coco()
随后我们看到这里调用了相关的函数转化为json文件需要的字典格式：
```python
	def data2coco(self):
		data_coco = {}
		data_coco['images'] = self.images
		data_coco['categories'] = self.categories
		data_coco['annotations'] = self.annotations
		return data_coco
```

最后进行一下json.dump就成功保存好json文件了。


## 转化代码最终改良版
经过上述描述，最终做一个改良，得到相应的结果：

```python
"""本模块用于批量转换labelme标记的格式，使之变成coco数据集格式"""
# -*- coding:utf-8 -*-
# !/usr/bin/env python


import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
import os


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class labelme2coco(object):

    def __init__(self, labelme_json=[], save_json_path='./tran.json'):  # 这里定义了默认的存储路径
        '''
        labelme_json: 所有labelme的json文件路径组成的列表
        save_json_path: json保存位置
        '''
        self.labelme_json = labelme_json     # labelme_json的文件路径，是一个列表存储着所有的待转化json文件路径
        self.save_json_path = save_json_path  # 保存的路径
        self.images = []  # 对应生成COCO形式json文件的image的类型。
        self.categories = []  # 同上
        self.annotations = []  # 同上
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.root = '/home/lry/projects/mmdetection/data/780b/'

        self.save_json()  # 注意这里一调用该类就直接运行相关的所有代码了，这个技巧很有用。

    def data_transfer(self):

        for num, json_file in enumerate(self.labelme_json):
            print("num:" + str(num + 1) + '    ' + json_file)
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                # 注意这个image是个函数！ 将该图片的所有相应的信息转化为COCO的image格式
                self.images.append(self.image(data, num))

                # shapes是主要的标注信息，我们重点就是读取这个，但是我们没有必要读取所有，可能只需要读取第一个780b就可以
                for shapes in data['shapes']:

                    # 首先我们会判断是否这个label是否是"780B"
                    if shapes['label'] != "780B":
                        continue

                    # 注意这里是一个循环，这个循环将所有的label有关文件都循环了一遍
                    label = shapes['label']
                    if label not in self.label:
                        # 注意这个self.categorie也是个函数，这个代码段就是要生成COCO的categories格式
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    points = shapes['points']
                    #points.append([points[0][0], points[1][1]])
                    #points.append([points[1][0], points[0][1]])
                    # 同样的，self.annotation也是一个相应的函数要将所给的points文件转化为annotation和segmentation文件
                    self.annotations.append(
                        self.annotation(points, label, num))
                    self.annID += 1


    def image(self, data, num):
        image = {}
        # 解析原图片数据，注意我们并不是所有文件都有,怪不得所有文件都打开运行会报错，真是好奇怪，但是我们可以通过文件路径打开，参见下一行代码
        try:
            img = utils.img_b64_to_arr(data['imageData'])
        except:
            img=io.imread(os.path.join(self.root, data['imagePath'])) # 通过图片路径打开图片， 不过这个就需要设一个全局的root
        # img = cv2.imread(data['imagePath'], 0)
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        # image['file_name'] = data['imagePath'].split("\\")[-1] #此行不适合我的数据集
        image['file_name'] = data['imagePath'].split("\\")[-1].split('..')[-1]

        self.height = height
        self.width = width

        return image

    def categorie(self, label):
        categorie = {}
        # categorie['supercategory'] = 'Cancer'  # 这个无用
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = label
        return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = self.getcatid(label)  # 注意，源代码默认为1
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        # print(polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        # print(index)

        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        # print(rows)
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'),
                  indent=4, cls=MyEncoder)  # indent=4 更加美观显示


# labelme_json = glob.glob(
#     '/home/lry/projects/mmdetection/data/780b/*.json')
# print(labelme_json)


# # labelme_json=['./Annotations/*.json']
# # labelme_json=glob.glob('./Annotations/*.json')

# saveme_json = '/home/lry/projects/mmdetection/lry/draft.json'
# print("Start.")

# labelme2coco(labelme_json, saveme_json)
# print("Finished.")


val_json = glob.glob('/home/lry/data/780b_std/val/*.json')
print(val_json)
saveme_json = '/home/lry/data/780b_std/val/annotation_coco.json'
print("Start.")

labelme2coco(val_json, saveme_json)
print("Finished.")

train_json = glob.glob('/home/lry/data/780b_std/train/*.json')
print(train_json)
saveme_json = '/home/lry/data/780b_std/train/annotation_coco.json'
print("Start.")

labelme2coco(train_json, saveme_json)
print("Finished.")
```

## 划分训练集和验证集
这里按照7：3来划分训练集和验证集:
```python
"""本模块用于划分验证集和测试集，大概比例为7：3"""

import shutil
import os
import random
import glob

def move_file(old_path, imgs,  new_path):
    '''这个函数是用来移动图片和相应的json文件'''
    print(old_path)
    print(new_path)
    # filelist = os.listdir(old_path) #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
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
```

## 设定config文件
我们决定用mask_rcnn，这里直接在`configs`目录下创建一个文件夹叫做fenghuo，然后加入一个`mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py`文件就OK。文件内容为：
```python
"""本模块用于放置configs，定义模型结构,
该配置文件将放到`/home/lry/projects/mmdetection/configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py`"""

# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('780B',)
data = dict(
    train=dict(
        img_prefix='/home/lry/projects/mmdetection/data/780b_std/train/',
        classes=classes,
        ann_file='/home/lry/projects/mmdetection/data/780b_std/train/annotation_coco.json'),
    val=dict(
        img_prefix='/home/lry/projects/mmdetection/data/780b_std/val/',
        classes=classes,
        ann_file='/home/lry/projects/mmdetection/data/780b_std/val/annotation_coco.json'),
    test=dict(
        img_prefix='/home/lry/projects/mmdetection/data/780b_std/val/',
        classes=classes,
        ann_file='/home/lry/projects/mmdetection/data/780b_std/val/annotation_coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
```
## 训练模型
```python
python tools/train.py configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py
```

## 测试模型
```python
CUDA_VISIBLE=1 python tools/test.py configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo/latest.pth  --eval bbox segm
```



# 2021.11.14日早更新

* 远程仓库：https://www.jianshu.com/p/e4bb2ac7e770

目前需要新建远程GitHub仓库方便我与别人协作

目前整个大框的检测还比较成功，接着就是检测每个小框的分类任务。



# 2021.11.29日晚更新
远程仓库创建成功，仓库地址：[烽火检测](https://github.com/LRY89757/mmdet-for-fenghuo)

目前需要做的事情：
* 写一个关于这类Labelme文档需要的dataloader，写一个尽量比较全面适配性比较强、可迁移性比较好的dataloader
* 对于运行出来的结果的图像做一个简单的切分，平均分为19份然后将每一个单独的盘都crop出来然后将非盘区域标黑放到resnet101等各类分类网络中进行相关的分类任务。
* 划分测试集和训练集的时候选择使用生成val.txt还有train.txt文件的格式，每个.txt文件内防止相关数据的路径。而不是将其复制到一个新的文件夹中进行分类，万一硬盘没有那么大的空间怎么办，不要滥用硬盘资源。



# 2021.11.30日晚更新
目前已经确定本周先做第二项，就是做一下目前盘符的分类操作。
（实际上目前这个任务的主要步骤之类的还不是特别清晰，还是需要进一步与组长讨论交流，不过关于cv2的一些操作这类可以自己先学一学，反正具体的内容知道了，更具体的细节还没沟通好。）



## The result of `inference_detector()`

目前第一步遇到的困难就是暂时不知道我们`inference_detector()`这个函数得到的结果的大致形式。因为之前做分割的时候直接将结果放到函数的参数中`model.show_result(img, result, out_file=f'/home/lry/projects/mmdetection/lry/demo{i}.jpg')`来显示结果了，再加上我本来对这个也不太熟悉，所以目前打算深入理解下关于函数`inference_detector()`，顺便将这一系列有关的都了解一下：`init_detector,inference_detector, show_result_pyplot`,这些都是源于库`mmdet.apis`,所以我们来找一下[官方文档](https://mmdetection.readthedocs.io/en/latest/api.html)关于mmdet.apis的有关解释：
实际上有关`inference_detector`的[内容](https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.apis.inference_detector)十分有限：

> mmdet.apis.inference_detector(*model*, *imgs*)[[SOURCE\]](https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/apis/inference.html#inference_detector)
>
> Inference image(s) with the detector.
>
> - Parameters
>
>   **model** (*nn.Module*) – The loaded detector.**imgs** (*str/ndarray* *or* *list**[**str/ndarray**] or* *tuple**[**str/ndarray**]*) – Either image files or loaded images.
>
> - Returns
>
>   If imgs is a list or tuple, the same length list type results will be returned, otherwise return the detection results directly.

当然感兴趣可以去选择看一下有关源码.这里我选择用代码来探索下有关的结果:
```python
from mmdet.apis import init_detector,inference_detector, show_result_pyplot
import mmcv
import os
import glob
from PIL import Image
from torchvision.transforms import ToTensor

config_file = "/home/lry/projects/mmdetection/configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py"
checkpoint_file = "/home/lry/projects/mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo/latest.pth"
model = init_detector(config_file, checkpoint_file,device="cuda:3")

img = '/home/lry/projects/mmdetection/data/780b_std/IMG_20211021_153945.jpg'
result = inference_detector(model, img)
print(result, '\n\n', result[0], '\n\n', result[0][0])
print(result[1][0][0].shape)
img = Image.open(img)
print(ToTensor()(img).shape)

```

有关这部分主要就看result的输出到底是什么东西,如果我们想要保存结果到某地或是展示结果可以用`model.show_result(img, result, out_file=f'/home/lry/projects/mmdetection/lry/demo{i}.jpg')`来显示.

简单来看下代码运行结果:
```python
([array([[3.6979218e+02, 2.3255410e+03, 2.7979324e+03, 4.2280156e+03,
        9.9831665e-01]], dtype=float32)], [[array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])]]) 

 [array([[3.6979218e+02, 2.3255410e+03, 2.7979324e+03, 4.2280156e+03,
        9.9831665e-01]], dtype=float32)] 

 [[3.6979218e+02 2.3255410e+03 2.7979324e+03 4.2280156e+03 9.9831665e-01]]
(4608, 3456)
torch.Size([3, 4608, 3456])
```
可以看到,result是一个元组,元组的第一个元素是一个np.array,含有5个元素,5个元素中第一个代表的是检测框的准确度,然后四个参数代表的是图片的检测框的左上右下相关参数,然后第二个元素就是图片分割数据集的分割结果了.这个结果是由True和False组成的和图像本身Tensor大小相同的一个矩阵.(True代表该像素是含有物体的)。

## Code Trace of `mmdet.apis.inference_detector()`
的确刚才我们将这个接口当作黑盒来探索到了他的一些具体功能。这样对于做项目来说非常足够了，但是我们这里深入一下来探索一下他的源码是怎么样的（~~感谢蒋哥带我飞~~）。

首先来看我们的inference_detector()源码`mmdet/apis/inference.py`:
```python
def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results
```

实际上这么长的源码，真正关键的就倒数第六行的代码`results = model(return_loss=False, rescale=True, **data)`这行代码调用了model来进行推理然后返回了结果results。我们要看model的而model是我们传进来的参数，是我们使用`init_detector`这个接口来调用的，这个接口使用了我们的config文件以及相关的checkpoint文件。这里config文件已经非常熟悉了，就是利用这个搭建的模型。我们需要看的是关于config文件里面所用的模型。而要最后的输出是由`roi_heads`，也就是maskrcnn最后一个模块来输出的。我们先追踪一下有关的config文件：
首先是我们自己定义的上层的config文件`configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py`, 从这里找可以看到我们继承的是`_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'`那么我们继续找到这个模块就`configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py`,可以看到他继承的是`_base_ = './mask_rcnn_r50_fpn_1x_coco.py'`,我们继续找这个文件`configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py`,这个文件有点像是一个package的`__init__.py`，就是代码只有所有需要继承的基本类型：
```python
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```
这里就是一些模型具体配置的继承基类文件，我们想要看模型的`roi_heads`部分，那么我们就去找`'../_base_/models/mask_rcnn_r50_fpn.py'`文件。我们可以在`configs/_base_/models/mask_rcnn_r50_fpn.py`里找到`roi_heads`的定义:
```python
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
```
这里面可以看到整个`roi_head`都是属于`type='StandardRoIHead'`, 由于MMDetection的注册器Registry机制，我们知道我们需要去到`mmdet`文件夹中去找这些东西，在`mmdet/models/roi_heads`中我们可以找到`mmdet/models/roi_heads/standard_roi_head.py`这个文件。
然后我们需要从类中寻找关于forward的有关函数，刚开始找的是`forward_dummy()`这个函数，但是从这个函数中看不到关于cls_score和bbox_pred以及mask_pred的合并有关步骤，或者说不太能对应上，后来打了一些断点发现似乎程序根本不会经过这个函数。所以应该不是经过这个forward函数，而另外的forward函数，首先可以排除`forward_train`这个函数，或者是排除所有和train有关的函数，因为我们毕竟这个是做的推断，所以我们需要的不是去train，至少也在test里面。

后来经过打断点发现程序运行在`simple_test()`这个函数里面：
```python
    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
```

接着调试发现这里调用了函数bbox2result()来实现了关于转化的一些问题。

实际上到此就结束了，当然`forward_dummy()`这个函数是非常有研究意义的，在`mmdet/models/detectors/two_stage.py`里可以看到这个是为了，特别是这里调用了`_bbox_forward()`这个函数,值得注意一下。


另外我们以上仅仅是讲了一个mask_rcnn的一个很小的`roi_head`部分，如果要总体看全体的一个概况的话，可以选择去看一下`mmdet/models/detectors/two_stage.py`虽然也不算是总体，但是也高了一级，可以看一下具体的这个`TwoStageDetector`是怎么调度的，是如何运行起来的。




## dataloader in mmdetection

其次是关于这周要做的第一个任务，这个任务实际上就是来做一个Labelme数据的dataloader,当然说起来容易，这个要把dataloader融到MMDetection里面还是需要深入理解他的Rigistry机制的，而且光写代码这一步如果完全不看他的关于COCODataset的代码靠自己写恐怕是非常难办的。所以我们这里一步一步的来探索。

首先我们来找到COCO的dataloader到底在哪里。

还是从config文件开始找，由于上文已经找过了，这里直接引用上文的寻找过程：


""""
首先是我们自己定义的上层的config文件`configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fenghuo.py`, 从这里找可以看到我们继承的是`_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'`那么我们继续找到这个模块就`configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py`,可以看到他继承的是`_base_ = './mask_rcnn_r50_fpn_1x_coco.py'`,我们继续找这个文件`configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py`,这个文件有点像是一个package的`__init__.py`，就是代码只有所有需要继承的基本类型：
```python
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```
"""

然后我们可以看到在`'../_base_/datasets/coco_instance.py'`里面，前几行代码就有`dataset_type = 'CocoDataset'`和上文相同的Registry机制，我们到`mmdet`文件夹中去寻找，可以发现在`mmdet/datasets/coco.py`里面有`CocoDataset`这个类。这个就是关于CocoDataset的dataloader.终于找到了！

但是有574行代码……………………蚌埠住了😥。

# 2021.12.1日上午更新

## dataloader in mmdet
再稍微说一下关于Rigistry还有继承类需要做的一些东西。

## 图像分割盘符切片数据预处理操作

### Preface
昨天本来打算认真做一下关于盘符分类的任务的，没想到还是最终没有弄，光顾着搞MMdetection的源码这些东西了没有来得及搞，目前已知接口`inference_detector`返回的result的结果了，就是一个元组tuple，第一个元素是检测框的位置和置信度，然后第二个元素就是分割的结果，是一个与图片大小相同的一个二维矩阵，由True和False组成。

我们接下来开始尝试做图像的处理。

首先构建detector得到一些具体结果：
![](https://s6.jpg.cm/2021/12/01/LqL6m6.png)

### 图像处理算法分析与算法步骤实现


#### 对于分割的结果预处理
首先，这个检测框的结果并不是非常靠谱,很明显如果我们选择对得到的bbox来进行均等分割是完全不行的，分割的结果还要好一些相对而言，所以我们这里可以选择考虑一下使用分割数据但是要对分割数据进行一些转换或者是说修补。

目前一个想法是对于分割结果的边界取平均值来得到4条可以用的边组成一个新的bbox。然后我们再对得到的新的bbox进行平均切分成19等份，当然这里没有考虑是否是倾斜的，因为我们要切出的图片要都是水平矩形，然后根据坐标关系将我们切除矩形部分不包含盘符的部分涂黑（所以写代码时不妨先将非盘符部分涂黑。）

目前考虑的算法步骤是：
* 根据构建detector得到的分割结果矩阵，根据边界平均思想得到一个 bbox
* 得到bbox的四个顶点（一定是bbox矩阵最靠左，最靠右，最靠上，最靠下的四个顶点）。
* 根据四个顶点的关系得到左右的两组边的点，然后根据左右等比例关系标注好中间要切分的18个点，连同着左右两端点都放到一个列表里，上下都要有，（其实也就通过给bbox的矩阵上下两条边等分成19份，以利于后续画图切割。）
* 开始切割，并选择将图片中非盘符的部分涂黑（这个可以在切割之前就涂黑了）
* 对一张图片得到的19个盘符标上种类，至于种类可以选择利用当时的json文件来标种类。

细节补充：
* 首先通过四个顶点得到一个水平bbox可以写一个函数复用
* 给定两个顶点将其连线等分割成19份并返回该直线所分坐标列表，或者说给定四个顶点返回切分好的各个小盘符顶点坐标列表（意思就是列表中每个元素都是一个分好的小框的四个顶点坐标）。
* 涂黑操作定义一个函数，输入为
* **以上所有的操作都放到一个类里统一处理，然后到时候调用直接构建类的时候就调用了进行完预处理操作。**


#### 对于普通标注labelme文件的结果预处理

因为Labelme文件是对于每一个图片都有一个单独的json文件，然后每一个json文件都会有各个盘符的位置信息还有具体的坐标种类，所以这样一来相比于分割的结果要好很多，所以我们这里思路简单很多：
* 读取`.json`文件
* 对于将每个小盘符四个顶点转化为一个较大的bbox
* 大框bbox内非盘符部分涂黑
* 保存，注意图片标签的保存


### 普通labelme文件读取分析



