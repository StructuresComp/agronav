import os.path as osp
import sys 

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

# define class and palette for agronav
classes = ('soil', 'sidewalk', 'vegetation', 'sky', 'human', 'vehicle', 'building', 'wall', 'others')
palette = [[128, 64, 128], [244, 35, 232], [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142], [70, 70, 70], [102, 102, 156], [0, 0, 0]]

data_root = osp.abspath(osp.join(osp.dirname(__file__), 'data/agronav/train/'))
img_dir = 'images'
ann_dir = 'labels'

@DATASETS.register_module()
class AgroNavDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.JPG', seg_map_suffix='.png', 
                     split=split, reduce_zero_label=False, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None