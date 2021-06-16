
import os
import numpy as np 
import pandas as pd 
from datetime import datetime
import time
import random
from tqdm import tqdm_notebook as tqdm # progress bar
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

# torchvision
import torchvision

# sklearn
from sklearn.model_selection import StratifiedKFold

# CV
import cv2

# Albumenatations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

#from pycocotools.coco import COCO
from sklearn.model_selection import StratifiedKFold

# glob
from glob import glob

# numba
import numba
from numba import jit

import warnings

    
 device = torch.device(DefaultConfig['device']) if torch.cuda.is_available() else torch.device('cpu')   
 
## Choose your optimizers:
Adam = False
if Adam: 
    Adam_config = {"lr" : 0.001, "betas" : (0.9, 0.999), "eps" : 1e-08}
else:
    SGD_config = {"lr" : 0.001, "momentum" : 0.9, "weight_decay" : 0.001}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
# seed_everything(DefaultConfig['seed'])


def get_train_transforms():
# data augmetnation for training dataset
    return A.Compose([
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                 val_shift_limit=0.2, p=0.3), 
            A.RandomBrightnessContrast(brightness_limit=0.2,  
                                       contrast_limit=0.2, p=0.3),
        ], p=0.2),
        A.Resize(height=DefaultConfig['img_size'], width=DefaultConfig['img_size'], p=1.0),
        #A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ],
    p=1.0, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transforms():
# data tranformation for validation dataset
    return A.Compose([
        A.Resize(height=DefaultConfig['img_size'], width=DefaultConfig['img_size'], p=1.0),
        #A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    



### Data Loader for the object detection (include bounding boxes)

class BoxingLungDataset(Dataset):

    def __init__(self, image_ids, df, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.df = df
        self.file_names = df['jpg_path'].values
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        
        image, boxes, labels = self.load_image_and_boxes(index)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    break
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(self.file_names[index], cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['id'] == image_id]       
        boxes = []
        for bbox in records[['frac_xmin', 'frac_ymin', 'frac_xmax', 'frac_ymax']].values:
            bbox = np.clip(bbox, 0, 1.0)
            temp = A.convert_bbox_from_albumentations(bbox, 'pascal_voc', image.shape[0], image.shape[0]) 
            boxes.append(temp)
        '''
        [0: 'atypical', 1: 'indeterminate', 2: 'negative', 3: 'typical']
        '''
        labels = records['integer_label'].values
        return image, boxes, labels


def get_train_dataset(fold_number):    
    return CustomDataset(
        image_ids = df_folds[df_folds['fold'] != fold_number].index.values,
        df = train,
        transforms = get_train_transforms()
    )

def get_validation_dataset(fold_number):
    return CustomDataset(
        image_ids = df_folds[df_folds['fold'] == fold_number].index.values,
        df = train,
        transforms = get_valid_transforms()
    )

def get_train_data_loader(train_dataset, batch_size=16):
    return DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4,
        collate_fn = collate_fn
    )

def get_validation_data_loader(valid_dataset, batch_size=16):
    return DataLoader(
        valid_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4,
        collate_fn = collate_fn
    )    

def collate_fn(batch):
    return tuple(zip(*batch))










