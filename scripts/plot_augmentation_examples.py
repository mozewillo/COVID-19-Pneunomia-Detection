import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
import cv2


if __name__ == '__main__':
    train_image = pd.read_csv('../data/train_image_level.csv')

    transform1 = A.Rotate(limit=30, p=1, border_mode=cv2.BORDER_CONSTANT)
    transform2 = A.RandomBrightnessContrast(brightness_limit=[-0.3,0.3], contrast_limit=[-0.3, 0.3], p=1)
    transform3 = A.ShiftScaleRotate(scale_limit=[-0.15,0.4], shift_limit=0, rotate_limit=0, p=1, border_mode=cv2.BORDER_CONSTANT)
    transform4 = A.ShiftScaleRotate(scale_limit=0, shift_limit=0.20, rotate_limit=0, p=1, border_mode=cv2.BORDER_CONSTANT)


    fig, axes = plt.subplots(3,5, figsize=(12,7))
    i = 0
    for img_id in train_image.id[:3]:
        img = cv2.imread('../data/train/'+img_id+'.png', 0)
        img1 = transform1(image=img)['image']
        img2 = transform2(image=img)['image']
        img3 = transform3(image=img)['image']
        img4 = transform4(image=img)['image']
        axes[i,0].imshow(img, cmap='gray')
        axes[i,1].imshow(img1, cmap='gray')
        axes[i,2].imshow(img2, cmap='gray')
        axes[i,3].imshow(img3, cmap='gray')
        axes[i,4].imshow(img4, cmap='gray')
        i+=1
        
    axes[0,0].set_title('Original image', size=13)
    axes[0,1].set_title('Random rotation', size=13)
    axes[0,2].set_title('Modified brightness', size=13)
    axes[0,3].set_title('Scaling', size=13)
    axes[0,4].set_title('Shifting', size=13)
    axes[0,0].set_ylabel('Image 1', size=13)
    axes[1,0].set_ylabel('Image 2', size=13)
    axes[2,0].set_ylabel('Image 3', size=13)
    [axi.get_xaxis().set_ticks([]) for axi in axes.ravel()]
    [axi.get_yaxis().set_ticks([]) for axi in axes.ravel()]
    plt.savefig('../plots/augmentations.png', dpi=150, bbox_inches='tight')
    plt.close()