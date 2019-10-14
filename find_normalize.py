import cv2
import numpy as np
import os
import glob

def main():
    IMG_DIR = './dataset/train'
    img_names = glob.glob(os.path.join(IMG_DIR,'*','*.jpg'))
    # print(len(img_names))
    for i,name in enumerate(img_names):
        img = cv2.imread(name, 0)
        img = img.flatten()
        if i == 0:
            all_img = img.copy()
        else:
            all_img = np.concatenate((all_img, img), axis=0)
        if i == 100: break
    pass
    print(all_img.shape)
    mean = all_img.mean()
    std = all_img.std()
    print(mean)
    print(std)
    img = cv2.imread(img_names[0], 0)[100:110,100:110]
    print(img)
    img = (img - mean)/std
    print(img)

if __name__ == '__main__':
    main()