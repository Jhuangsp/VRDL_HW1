from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input as preprocess_renet50
from keras.applications.vgg16 import preprocess_input as preprocess_vgg16
import os
import glob
import numpy as np
import argparse
import csv
from cv2 import resize, INTER_CUBIC
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Inference', fromfile_prefix_chars='@')
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Choose model. (resnet or vgg)')
args = parser.parse_args()

DATASET_PATH = './dataset'
imgs_name = glob.glob(os.path.join(DATASET_PATH, 'test', '*.jpg'))
print('There are total {} images to test'.format(len(imgs_name)))

# net = load_model('model-resnet50-final.h5') if args.model == 'resnet' else load_model('model-vgg16-final.h5')
net = load_model('model-resnet50-final_best.h5') if args.model == 'resnet' else load_model('model-vgg16-final_best.h5')

class_list = ['bedroom', 'coast', 'forest', 'highway', 'insidecity', 'kitchen',
              'livingroom', 'mountain', 'office', 'opencountry', 'street', 'suburb', 'tallbuilding']

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

with open('answer.csv', 'w', newline='') as f:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
    writer.writeheader()

    for f in imgs_name:
        img = image.load_img(f, target_size=(200, 200))
        # img = image.load_img(f)
        if img is None:
            continue
        x = image.img_to_array(img)
        x = preprocess_renet50(x) if args.model == 'resnet' else preprocess_vgg16(x)
        x = np.expand_dims(x, axis=0)
        pred = net.predict(x)[0]
        # top_inds = pred.argsort()[::-1][:5]
        # print(f)
        # for i in top_inds:
        #     print('    {:.3f}  {}'.format(pred[i], class_list[i]))

        name = f.split(os.sep)[-1].split('.')[0]
        answer = class_list[pred.argsort()[::-1][0]]

        writer.writerow({fieldnames[0]: name, fieldnames[1]: answer})
