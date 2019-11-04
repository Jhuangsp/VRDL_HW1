import pprint
import datetime
import numpy as np
from hyperopt import hp, fmin, rand, tpe, space_eval
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pp = pprint.PrettyPrinter(indent=4)

def go(args):
    drop_rate = args[0]

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
                                                        target_size=(200, 200),
                                                        interpolation='bicubic',
                                                        class_mode='categorical',
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        subset='training')

    validation_generator = train_datagen.flow_from_directory(os.path.join(DATASET_PATH, 'train'),
                                                             target_size=(
                                                                 200, 200),
                                                             interpolation='bicubic',
                                                             class_mode='categorical',
                                                             batch_size=BATCH_SIZE,
                                                             subset='validation')  # set as validation data

    net = VGG16(include_top=False, weights='imagenet', input_tensor=None,
                input_shape=(200, 200, 3))
    x = net.output
    x = Flatten()(x)
    x = Dropout(drop_rate)(x)

    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(drop_rate)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    output_layer = Dense(NUM_CLASS, activation='softmax', name='softmax')(x)

    net_final = Model(inputs=net.input, outputs=output_layer)

    for layer in net_final.layers[:-4]:
        layer.trainable = False
    for layer in net_final.layers[-4:]:
        layer.trainable = True

    net_final.compile(optimizer=Adam(lr=1e-4),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    mcp_save = ModelCheckpoint(os.path.join(SAVE_PATH, 'model-vgg16-final_best.h5'), monitor='val_loss',
                               verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    print(net_final.summary())
    history = net_final.fit_generator(train_generator,
                                      steps_per_epoch=train_generator.samples // BATCH_SIZE,
                                      validation_data=validation_generator,
                                      validation_steps=validation_generator.samples // BATCH_SIZE,
                                      epochs=NUM_EPOCHS,
                                      callbacks=[mcp_save],
                                      verbose=1)

    net_final.save(os.path.join(SAVE_PATH, 'model-vgg16-final.h5'))
    with open(os.path.join(SAVE_PATH, 'min_val_loss.txt'), 'a') as out_file:
        out_file.write(str(min(history.history['val_loss'])))
    print('Loss:', min(history.history['val_loss']))

    return min(history.history['val_loss'])



loss = go([0.25])
# print(loss)
