import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import pudb

# Set up paths
# data_root = 'data/cityscape-freiburg'
# data_root = 'data/agronav/train/vit-inferences/'
# data_root = 'data/agronav/train/manual/'
data_root = 'data/train/'
img_dir = 'images'
ann_dir = 'labels'

# define class and plaette for better visualization
# classes = ('soil', 'sidewalk', 'vegetation', 'sky', 'human', 'vehicle', 'others')
# palette = [[19, 156, 181], [244, 35, 232], [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142], [0, 0, 0]]

classes = ('soil', 'sidewalk', 'vegetation', 'sky', 'human', 'vehicle', 'building', 'wall', 'others')
palette = [[128, 64, 128], [244, 35, 232], [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142], [70, 70, 70], [102, 102, 156], [0, 0, 0]]

# split train/val set randomly
split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    # select first 4/5 as train set
    train_length = int(len(filename_list)*4/5)
    f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
    # select last 1/5 as val set
    f.writelines(line + '\n' for line in filename_list[train_length:])

@DATASETS.register_module()
class AgroNavDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.JPG', seg_map_suffix='.png', 
                     split=split, reduce_zero_label=False, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

from mmcv import Config

# cfg = Config.fromfile('configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py')
# cfg = Config.fromfile('configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py')
# cfg = Config.fromfile('configs/resnest/fcn_s101-d8_512x1024_80k_cityscapes.py')
cfg = Config.fromfile('configs/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes.py')

from mmseg.apis import set_random_seed

# Since we use ony one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg

# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 9

# cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# cfg.model.auxiliary_head.num_classes = 9

# for i in range(len(cfg.model.auxiliary_head)):
#     cfg.model.auxiliary_head[i].norm_cfg = cfg.norm_cfg
#     cfg.model.auxiliary_head[i].num_classes = 9


# Modify dataset type and path
cfg.dataset_type = 'AgroNavDataset'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 6
cfg.data.workers_per_gpu= 1

# cfg.img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# # pu.db
# # cfg.crop_size = (256, 512)
# cfg.train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=(2048, 1024),  keep_ratio=True),
#     dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **cfg.img_norm_cfg),
#     dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]

# cfg.test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 1024),
#         img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
#         flip=True,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **cfg.img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]


cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = 'checkpoints/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth'
# cfg.load_from = 'checkpoints/fcn_hr18s_512x1024_40k_cityscapes_20200601_014216-93db27d0.pth'
# cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
# cfg.load_from = 'checkpoints/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth'
# cfg.load_from = 'checkpoints/lraspp_m-v3-d8_512x1024_320k_cityscapes_20201224_220337-cfe8fb07.pth'
# cfg.load_from = 'checkpoints/fcn_s101-d8_512x1024_80k_cityscapes_20200807_140631-f8d155b3.pth'

# cfg.load_from = 'output/ResNest-FCN/iterations-40000/iter_40000.pth'
# cfg.load_from = 'output/HRNet/iterations-40000/iter_40000.pth'
# cfg.load_from = 'output/MobileNetV3/iterations-40000/iter_40000.pth'

# cfg.load_from = 'output/ResNest-FCN/agronav/iterations-40000/iter_40000.pth'
# cfg.load_from = 'output/HRNet/agronav/iterations-40000/iter_40000.pth'
# cfg.load_from = 'output/MobileNetV3/agronav/iterations-40000/iter_40000.pth'

cfg.load_from = 'checkpoints/MobileNetV3.pth'
# cfg.load_from = 'output/HRNet/agronav/iterations-40000-manual/iter_30000.pth'
# cfg.load_from = 'output/MobileNetV3/agronav/iterations-40000-manual/iter_30000.pth'



# cfg.resume_from = 'output/HRNet/iterations-40000/iter_40000.pth'
# cfg.resume_from = 'output/MobileNetV3/iterations-40000/iter_40000.pth'
# cfg.resume_from = 'output/ResNest-FCN/agronav/iterations-40000-manual/iter_15000.pth'
# cfg.resume_from = 'output/HRNet/agronav/iterations-40000-manual/iter_15000.pth'
# cfg.resume_from = 'output/MobileNetV3/agronav/iterations-40000-manual/iter_15000.pth'

# Set up working dir to save files and logs.
# cfg.work_dir = 'output/MobileNetV3/iterations-40000'
# cfg.work_dir = 'output/ResNest-FCN/iterations-40000'
# cfg.work_dir = 'output/HRNet/iterations-40000'

cfg.work_dir = 'output'
# cfg.work_dir = 'output/HRNet/agronav/iterations-40000-total/'
# cfg.work_dir = 'output/MobileNetV3/agronav/iterations-40000-total/'

cfg.log_config.interval = 10
cfg.evaluation.interval = 5000
cfg.checkpoint_config.interval = 5000

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
cfg.runner.max_iters = 15000

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
