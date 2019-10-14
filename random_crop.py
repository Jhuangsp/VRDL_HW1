import pprint
import datetime
import numpy as np
import glob
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pp = pprint.PrettyPrinter(indent=4)


def crop(img):
    shape = img.shape
    if shape[0] > shape[1]:
        extra = shape[0] - shape[1]
        img = img[extra//2:extra//2+shape[1], :]
    elif shape[0] < shape[1]:
        extra = shape[1] - shape[0]
        img = img[:, extra//2:extra//2+shape[0]]
    assert img.shape[0] == img.shape[1], 'error'
    return img


now = datetime.datetime.now()
current_time = '{:04d}_{:02d}_{:02d}_{:02d}{:02d}{:02d}'.format(
    now.year, now.month, now.day,
    now.hour, now.minute, now.second)

DATASET_PATH = './dataset/train'
DATASET_PATH_NEW = './dataset/train_after_aug'

imgs = glob.glob(os.path.join(DATASET_PATH, '*', '*.jpg'))
print(len(imgs))

for name in imgs:
    tar_dir = os.path.join(DATASET_PATH_NEW,
                           name.split('\\')[-2],
                           name.split('\\')[-1])
    img = cv2.imread(name)
    img = crop(img)
    cv2.imwrite(tar_dir, img)