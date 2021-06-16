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
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    for i in range(1, 4):
        data = train_image[train_image.Class==i]['nr_of_boxes'].value_counts().sort_index()
        axes[i-1].bar(x=[str(i) for i in list(data.index)], height=data, width=0.9)
        axes[i-1].set_xlabel('Number of boxes', fontsize=12)
        axes[i-1].set_title(class_names[i])
        for rect in axes[i-1].patches:
            axes[i-1].text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                           "%.1f%%"% (rect.get_height()/data.sum()*100),
                           ha='center')
    axes[0].set_ylabel('Number of images', fontsize=12)
    plt.suptitle('Number of boxes on images from each class', size=18, y=1.05)
    plt.savefig('../plots/nr_of_boxes.png', dpi=150, bbox_inches='tight')
    plt.close()