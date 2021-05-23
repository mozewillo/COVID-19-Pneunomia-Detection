import numpy as np
import cv2


def remove_noise(img, min_size=10):
    height, width = img.shape

    # find all the connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # Removes background, which is seen as a big component
    sizes = stats[1:, -1]
  
    blob_idxs = []    
    for idx, s in enumerate(sizes):
        if s < min_size:
            blob_idxs.append(idx+1)
    
    img[np.isin(output, blob_idxs)] = 0

    return img


def crop(img):
    _, thresh = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    img_cropped = img[y_min:y_max, x_min:x_max]
    
    return img_cropped