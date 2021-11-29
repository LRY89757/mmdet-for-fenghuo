from mmcv import Config
import mmdet.apis

if __name__ == "__main__":
    cfg = Config.fromfile("configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py")
    # print(cfg)
    # print('cfg.model:\n', cfg.model)
    # print('cfg.train_cfg:\n', cfg.get('train_cfg'))
    # print('cfg.test_cfg:\n', cfg.get('test_cfg'))
    # print('cfg.data.train:\n', cfg.data.train)
    # print('cfg.data.train[\'type\']:\n', cfg.data.train['type'])
    
    # about build_dataset(cfg.data.train)
    print(isinstance(cfg.data.train, (list, tuple)))
    print(isinstance(cfg.data.train.get('ann_file'), (list, tuple)))
    print("hello world")
