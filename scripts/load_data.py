import pandas as pd
import numpy as np
import os
from PIL import Image
import pydicom
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir, walk
import torch
from skimage import exposure


TRAIN_DIR = '../input/siim-covid19-detection/train'
TEST_DIR = '../input/siim-covid19-detection/test'

train = pd.read_csv('../input/siim-covid19-detection/train_image_level.csv')
train_study = pd.read_csv('../input/siim-covid19-detection/train_study_level.csv')
train_study['id'] = train_study['id'].apply(lambda i: i.split('_')[0])

# handling studies with multiple images
multi = train.StudyInstanceUID.value_counts()[train.StudyInstanceUID.value_counts()>1]
to_remove = []
for study in multi.index:
    d = train[train['StudyInstanceUID']==study]
    tup = (d['boxes'].isna().sum(), len(d), d['StudyInstanceUID'].iloc[0])
    if tup[0]<tup[1]:
        to_remove += d['boxes'][d['boxes'].isna()].index.tolist()
train = train.drop(index=to_remove)


class CovLungDataset(Dataset):
    def __init__(self, dir_path, labels_data, new_size=(512,512)):
        self.COUNTER = 0
        self.dir_path = dir_path
        self.labels_data = labels_data
        self.new_size = new_size

    def __len__(self):
        return len(self.labels_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_dir = self.labels_data.iloc[idx]['id']
        img_label = self.labels_data.iloc[idx]['Negative for Pneumonia']
        
        path_to_img = os.path.join(self.dir_path, image_dir)
        path_to_img = os.path.join(path_to_img, listdir(path_to_img)[0]) 
        path_to_img = os.path.join(path_to_img, next(walk(path_to_img))[2][0])
        data = pydicom.dcmread(path_to_img)
        image = data.pixel_array
        image = exposure.equalize_hist(image)
        
        good_height, good_width = self.new_size
        image = cv2.resize(image, (good_width, good_height), interpolation=Image.LANCZOS)
        sample = {'image': image.reshape(1, good_height, good_width), 'label': img_label}
        return sample


transformed_dataset = CovLungDataset(dir_path=TRAIN_DIR,
                                     labels_data=train_study[['id', 'Negative for Pneumonia']],
                                     new_size=(424,512))   # or 512x512 or 256x256

dataloader = DataLoader(transformed_dataset, batch_size=32,
                        shuffle=False, num_workers=0)