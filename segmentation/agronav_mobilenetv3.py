import os.path as osp
import sys 

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
# from agronav_dataset import AgroNavDataset
import pudb

# Set up paths
# data_root = 'data/agronav/train/'
data_root = osp.abspath(osp.join(osp.dirname(__file__), 'data/agronav/train/'))
img_dir = 'images'
ann_dir = 'labels'

# define class and palette for agronav
classes = ('soil', 'sidewalk', 'vegetation', 'sky', 'human', 'vehicle', 'building', 'wall', 'others')
palette = [[128, 64, 128], [244, 35, 232], [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142], [70, 70, 70], [102, 102, 156], [0, 0, 0]]

@DATASETS.register_module()
class AgroNavDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.JPG', seg_map_suffix='.png', 
                     split=split, reduce_zero_label=False, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


from mmcv import Config
# cfg = Config.fromfile('configs/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes.py')
cfg_path = osp.abspath(osp.join(osp.dirname(__file__), 'configs/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes.py'))
cfg = Config.fromfile(cfg_path)

from mmseg.apis import set_random_seed

# Since we use ony one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg

# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 9

# Modify dataset type and path
cfg.dataset_type = 'AgroNavDataset'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 6
cfg.data.workers_per_gpu= 1

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

cfg.load_from = osp.abspath(osp.join(osp.dirname(__file__), 'checkpoints/MobileNetV3.pth'))
cfg.work_dir = osp.abspath(osp.join(osp.dirname(__file__), 'output'))

cfg.log_config.interval = 10
cfg.evaluation.interval = 5000
cfg.checkpoint_config.interval = 5000

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
cfg.runner.max_iters = 15000

cfg_mobilenetv3 = cfg.copy()

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
