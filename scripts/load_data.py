import pandas as pd
import pylab
import matplotlib as plt
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import preprocessing


labels = pd.read_csv('/content/drive/MyDrive/ADP/train_labels.csv', dtype={'image_id':'str',	'InChI':'str'})
labels = labels[labels['image_id'].str.startswith('0')]
labels['InChI'] = [s[9:] for s in labels['InChI']]     # removing 'InChI=1S/'
labels = labels.sample(frac=1)
n = len(labels)
val_n = int(n/10)
labels_val = labels.iloc[:val_n]
labels_train = labels.iloc[val_n:]


class ChemicalDataset(Dataset):
    def __init__(self, dir_path, labels_data, only_resize=False):
        self.dir_path = dir_path
        self.labels_data = labels_data
        self.only_resize = only_resize

    def __len__(self):
        return len(self.labels_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.labels_data.iloc[idx,0]
        img_label = self.labels_data.iloc[idx,1]
        img_dir = '/'.join(list(img_id[:3]))
        image = cv2.imread(self.dir_path + img_dir + '/'+ img_id + '.png', 0)
        image = ~image.astype('bool')
        image = image.astype('uint8')
        image = preprocessing.remove_noise(image, 2)
        image = preprocessing.crop(image)
        height, width = image.shape
        if height > width:
                image = np.rot90(image)
        good_height, good_width = 300, 500
        if height<=good_height and width<=good_width and not self.only_resize:
            image_padded = np.zeros((good_height,good_width))
            i = int((good_height-height)/2)
            j = int((good_width-width)/2)
            image_padded[i:i+height, j:j+width] = image
        else:
            ratio = width / height
            good_ratio = good_width / good_height
            if ratio < good_ratio:    # padding in width
                new_width = int(np.round(height * good_ratio))
                image_padded = np.zeros((height,new_width))
                j = int((new_width-width)/2)
                image_padded[:, j:j+width] = image
            else:    # padding in height
                new_height = int(np.round(width / good_ratio))
                image_padded = np.zeros((new_height,width))
                i = int((new_height-height)/2)
                image_padded[i:i+height,:] = image
            image_padded = cv2.resize(image_padded, (good_width,good_height), interpolation=cv2.INTER_NEAREST)
        sample = {'image': image_padded, 'label': img_label}
        return sample

transformed_dataset = ChemicalDataset(dir_path='/content/drive/MyDrive/ADP/',
                                      labels_data=labels_train,
                                      only_resize=False)

dataloader = DataLoader(transformed_dataset, batch_size=4,
                                                shuffle=True, num_workers=0)