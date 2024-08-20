import os.path as osp
import sys 

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from agronav_dataset import AgroNavDataset
import pudb

# Set up paths
data_root = osp.abspath(osp.join(osp.dirname(__file__), 'data/agronav/train/'))
img_dir = 'images'
ann_dir = 'labels'

from mmcv import Config
cfg_path = osp.abspath(osp.join(osp.dirname(__file__), 'configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py'))
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

cfg.load_from = osp.abspath(osp.join(osp.dirname(__file__), 'checkpoint/HRNet.pth'))
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

cfg_hrnet = cfg.copy()

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
