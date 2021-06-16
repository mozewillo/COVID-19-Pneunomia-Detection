import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

if __name__ == '__main__':
    train_image = pd.read_csv('../data/train_image_level.csv')
    train_study = pd.read_csv('../data/train_study_level.csv')
    train_study['id'] = train_study['id'].str.split('_').str[0]

    train_image['nr_of_boxes'] = (train_image.label.str.split().str.len()/6).astype(np.uint8)
    train_image.loc[train_image.label.str.startswith('none'), 'nr_of_boxes'] = 0

    classes = train_study.iloc[:,1:].idxmax(axis=1)
    classes[classes=='Negative for Pneumonia'] = 0
    classes[classes=='Typical Appearance'] = 1
    classes[classes=='Indeterminate Appearance'] = 2
    classes[classes=='Atypical Appearance'] = 3
    train_study['Class'] = classes
    train_study.index = train_study.id

    train_image = train_image.join(train_study['Class'], on='StudyInstanceUID')
    class_names = ['Negative for Pneumonia', 'Typical Appearance',
                   'Indeterminate Appearance', 'Atypical Appearance']

    ids = []
    classes = []
    boxes0 = []
    boxes1 = []
    boxes2 = []
    boxes3 = []

    for i, row in train_image.iterrows():
        boxes = row.label.split()
        for b in range(row.nr_of_boxes):
            ids.append(row.id)
            classes.append(row.Class)
            box = boxes[b*6+2:(b+1)*6]
            boxes0.append(box[0])
            boxes1.append(box[1])
            boxes2.append(box[2])
            boxes3.append(box[3])

    data_boxes = pd.DataFrame(index = range(len(ids)))
    data_boxes['image_id'] = ids
    data_boxes['Class'] = classes
    data_boxes['xmin'] = pd.to_numeric(boxes0)
    data_boxes['ymin'] = pd.to_numeric(boxes1)
    data_boxes['xmax'] = pd.to_numeric(boxes2)
    data_boxes['ymax'] = pd.to_numeric(boxes3)

    data_boxes['height'] = data_boxes['ymax']-data_boxes['ymin']
    data_boxes['width'] = data_boxes['xmax']-data_boxes['xmin']
    data_boxes['size'] = data_boxes['height'] * data_boxes['width']

    fig, axes = plt.subplots(1, 3, figsize=(15,5), sharex=True)
    for i in range(1, 4):
        data = data_boxes[data_boxes.Class==i]['height']
        axes[i-1].hist(data, bins=range(0,3201,200))
        axes[i-1].set_xlabel('Box height', fontsize=12)
        axes[i-1].set_title(class_names[i])

    axes[0].set_ylabel('Number of images', fontsize=12)
    plt.suptitle('Height of boxes for each class', size=18, y=1.05)
    plt.savefig('../plots/height_of_boxes.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15,5), sharex=True)
    for i in range(1, 4):
        data = data_boxes[data_boxes.Class==i]['width']
        axes[i-1].hist(data, bins=range(0,2001,200))
        axes[i-1].set_xlabel('Box width', fontsize=12)
        axes[i-1].set_title(class_names[i])

    axes[0].set_ylabel('Number of images', fontsize=12)
    plt.suptitle('Width of boxes for each class', size=18, y=1.05)
    plt.savefig('../plots/width_of_boxes.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15,5), sharex=True)
    for i in range(1, 4):
        data = data_boxes[data_boxes.Class==i]['size']
        axes[i-1].hist(data, bins=range(0,5400000,300000))
        axes[i-1].set_xlabel('Box area', fontsize=12)
        axes[i-1].set_title(class_names[i])

    axes[0].set_ylabel('Number of images', fontsize=12)
    plt.suptitle('Size of boxes for each class', size=18, y=1.05)
    plt.savefig('../plots/size_of_boxes.png', dpi=150, bbox_inches='tight')
    plt.close()