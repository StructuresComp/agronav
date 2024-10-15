# Create a custom torch dataset for row detection
from typing import List
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import json
from mmengine.registry import DATASETS, TRANSFORMS
from mmengine.dataset.base_dataset import Compose
import torch
from mmengine.dataset import BaseDataset
import copy
from mmengine.registry import FUNCTIONS
class RowDetectionDataset(BaseDataset):
   
   def __init__(self, dataset_dir, pipeline, filter_cfg=None, test_mode=False, max_refetch=1000):
       self.images_dir = os.path.join(dataset_dir, "images")
       self.labels_dir = os.path.join(dataset_dir, "labels")
       self.labels = os.listdir(self.labels_dir)
       self._indices = None
       self.filter_cfg = copy.deepcopy(filter_cfg)
       self.test_mode = test_mode
       self.max_refetch = max_refetch
       self.data_list: List[dict] = []
       self.data_bytes: np.ndarray
       self.serialize_data = False
       # Build pipeline.
       self.pipeline = Compose(pipeline)
   

   
   # def __len__(self):
   #     return len(self.labels)
   
   # def __getitem__(self, idx):
   #     label_name = self.labels[idx]
   #     with open(os.path.join(self.labels_dir, label_name), 'r') as f:
   #         label_file = json.load(f)
           
   #     img_name = f"{label_file['img_id']}.jpg"
   #     img = cv2.imread(os.path.join(self.images_dir, img_name))
   #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   #     labels = label_file['labels']
   #     label = labels[len(labels)//2]
   #     H, W, _ = img.shape
   #     x_alpha, y_alpha = self.fit_curves(label['x'], label['y'], label['alpha'], W, H)
   #     if(self.transform):
   #         # We need to apply tranform for the images and the labels
   #         img = self.transform(img)
   #         x_alpha = self.transform(x_alpha)
   #         y_alpha = self.transform(y_alpha)    
       
   #     return {
   #         "image": self.normalize(img),
   #         "targets": np.vstack([x_alpha, y_alpha])
   #     }
   
   def load_data_list(self) -> List[dict]:
       ret_list = []
       labels = os.listdir(self.labels_dir)
       for label in labels:
           obj = dict()
           img_path = os.path.join(self.images_dir, f'{label.split(".")[0]}.jpg')
           obj['img_path'] = img_path
           obj['label_path'] = os.path.join(self.labels_dir, label)
           ret_list.append(obj)
       return ret_list
   
   def normalize(self, img):
       return (img - 127.5)/127.5
   
   def fit_curves(self, x, y, alpha, W, H):
       x = np.array(x)
       y = np.array(y)
       alpha = np.array(alpha)
       x = x / W
       y = y / H
       alpha = alpha.astype(np.float32)
       x_alpha = np.polyfit(alpha, x, 3)
       y_alpha = np.polyfit(alpha, y, 3)
       return x_alpha, y_alpha

@FUNCTIONS.register_module()
def row_detection_metrics_collate(batch):
   images = [item['img'].transpose(2, 1) for item in batch]
   targets = [torch.tensor(item['targets']) for item in batch]
   key_points = [item['key_points'] for item in batch]
   ori_shapes = [item['ori_shape'] for item in batch]
   return {
       "images": torch.stack(images).float(),
       "targets": targets,
       "key_points": key_points,
       "ori_shapes": ori_shapes
   }


@FUNCTIONS.register_module()
def row_detection_collate(batch):
   images = [item['img'].transpose(2, 1) for item in batch]
   targets = [torch.tensor(item['targets']) for item in batch]
   return {
       "images": torch.stack(images).float(),
       "targets": targets,
   }


@DATASETS.register_module(name="RowDetection", force=False)
def build_row_detection_dataset(dataset_dir, pipeline):
   return RowDetectionDataset(dataset_dir, pipeline)
   
# Write tests for the dataset class use dataloader from pytorch

# Path: test_row_detection_dataset.py