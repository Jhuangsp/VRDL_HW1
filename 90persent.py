import pprint
import datetime
import numpy as np
from hyperopt import hp, fmin, rand, tpe, space_eval
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pp = pprint.PrettyPrinter(indent=4)

def normalize(img):
    img = np.array(img)
    mean = 129.88072497323765
    std = 62.47280975739285
    return (img - mean) / std

def go(args):
    drop_rate = args[0]
    # rotation_range = int(args[0]*30)
    # width_shift_range = args[1]*0.3
    # height_shift_range = args[2]*0.3
    # shear_range = args[3]*0.3
    # zoom_range = args[4]*0.5
    # if args[5] == 0:
    #     fill_mode = 'reflect'
    # elif args[5] == 1:
    #     fill_mode = 'wrap'
    # elif args[5] == 2:
    #     fill_mode = 'nearest'
    # elif args[5] == 3:
    #     fill_mode = 'constant'
    # else:
    #     print(error)
    #     os._exit(0)

    now = datetime.datetime.now()
    current_time = '{:04d}_{:02d}_{:02d}_{:02d}{:02d}{:02d}'.format(
        now.year, now.month, now.day,
        now.hour, now.minute, now.second)

    DATASET_PATH = './dataset'
    BATCH_SIZE = 20
    NUM_CLASS = 13
    NUM_EPOCHS = 20
    SAVE_PATH = current_time
    os.mkdir(SAVE_PATH)

    train_datagen = ImageDataGenerator(
        # rescale=1./255,
        # rotation_range=rotation_range,
        # width_shift_range=width_shift_range,
        # height_shift_range=height_shift_range,
        # shear_range=shear_range,
        # zoom_range=zoom_range,
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        fill_mode='reflect',
        cval=0,
        validation_split=0.1)

    train_generator = train_datagen.flow_from_directory(os.path.join(DATASET_PATH, 'train'),
                                                        target_size=(224, 224),
                                                        interpolation='bicubic',
                                                        class_mode='categorical',
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        subset='training')

    validation_generator = train_datagen.flow_from_directory(os.path.join(DATASET_PATH, 'train'),
                                                             target_size=(
                                                                 224, 224),
                                                             interpolation='bicubic',
                                                             class_mode='categorical',
                                                             batch_size=BATCH_SIZE,
                                                             subset='validation')  # set as validation data

    # print('Total batches:', train_generator.__len__())
    # itr = train_generator.__iter__()
    # batch = next(itr)
    # print('Input:', batch[0].shape, 'Output:', batch[1].shape)

    # for cls, idx in train_generator.class_indices.items():
    #     print('Class #{} = {}'.format(idx, cls))
    # os._exit(0)
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                   input_shape=(224, 224, 3), pooling='avg')
    # net = VGG19(include_top=False, weights='imagenet', input_tensor=None,
    #                input_shape=(224, 224, 3), pooling='max')
    x = net.output
    # x = Flatten()(x)
    x = Dropout(drop_rate)(x)

    x = Dense(256, name='D1', activation='elu')(x)
    output_layer = Dense(NUM_CLASS, activation='softmax', name='softmax')(x)
    net_final = Model(inputs=net.input, outputs=output_layer)

    for layer in net_final.layers[:-2]:
        layer.trainable = False
    for layer in net_final.layers[-2:]:
        layer.trainable = True

    net_final.compile(optimizer=Adam(lr=1e-4),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    mcp_save = ModelCheckpoint(os.path.join(SAVE_PATH, 'model-resnet50-final_best.h5'), monitor='val_loss',
                               verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # print(net_final.summary())
    history = net_final.fit_generator(train_generator,
                                      steps_per_epoch=train_generator.samples // BATCH_SIZE,
                                      validation_data=validation_generator,
                                      validation_steps=validation_generator.samples // BATCH_SIZE,
                                      epochs=NUM_EPOCHS,
                                      callbacks=[mcp_save],
                                      verbose=0)

    # pp.print(history.history)

    net_final.save(os.path.join(SAVE_PATH, 'model-resnet50-final.h5'))
    with open(os.path.join(SAVE_PATH, 'top_val.txt'), 'a') as out_file:
        out_file.write(str(max(history.history['val_acc'])))
    print(max(history.history['val_acc']))

    return 1-max(history.history['val_acc'])


# space = [hp.uniform('rotation_range', 0.0, 0.1),
#          hp.uniform('width_shift_range', 0.0, 0.1),
#          hp.uniform('height_shift_range', 0.0, 0.1),
#          hp.uniform('shear_range', 0.0, 0.1),
#          hp.uniform('zoom_range', 0.0, 0.1),
#          hp.randint('fill_mode', 4)]
space = [hp.uniform('drop_rate', 0.0, 0.5)]

# allloss = go((1/3, 1/3, 1/3, 1/3, 0.4, 0))

best = fmin(go, space, algo=tpe.suggest, max_evals=10)
print(best)
