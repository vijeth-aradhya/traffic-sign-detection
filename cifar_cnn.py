from __future__ import print_function
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import sys


batch_size = 32
num_classes = 43
epochs = 100
data_augmentation = True
train_dir = './GTSRB_Prev/Training/'
validation_dir = "./GTSRB_Prev/Validation/"
test_dir = "fill"
# num_train_images in ./GTSRB_Prev/Garbage = 2550
num_train_images = 26146
num_validation_images = 8054
num_test_images = "fill"
resize_dim = (100, 109)

# Create CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 109, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=False)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:

    train_generator = ImageDataGenerator(data_format="channels_last").flow_from_directory(
        train_dir,
        target_size=(resize_dim[0], resize_dim[1]), # Dimensions to which all the images will be resized
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    for i in range(0, 32):
        print(train_generator.filenames[i])

    print('\n')

    print('batch_index: {0}'.format(train_generator.batch_index))
    print('batch_size: {0}'.format(train_generator.batch_size))
    print('class_indices: {0}'.format(train_generator.class_indices))
    print('class_mode: {0}'.format(train_generator.class_mode))
    print('classes: {0}'.format(train_generator.classes))
    print('color_mode: {0}'.format(train_generator.color_mode))
    print('data_format: {0}'.format(train_generator.data_format))
    print('dir: {0}'.format(train_generator.directory))
    print('image_shape: {0}'.format(train_generator.image_shape))
    print('target_size: {0}'.format(train_generator.target_size))
    print('total_batches_seen: {0}'.format(train_generator.total_batches_seen))

    print('\n')

    validation_generator = ImageDataGenerator(data_format="channels_last").flow_from_directory(
        validation_dir,
        target_size=(resize_dim[0], resize_dim[1]),
        batch_size=batch_size,
        class_mode='categorical')

    print('\n')

    print('batch_index: {0}'.format(validation_generator.batch_index))
    print('batch_size: {0}'.format(validation_generator.batch_size))
    print('class_indices: {0}'.format(validation_generator.class_indices))
    print('class_mode: {0}'.format(validation_generator.class_mode))
    print('classes: {0}'.format(validation_generator.classes))
    print('color_mode: {0}'.format(validation_generator.color_mode))
    print('data_format: {0}'.format(validation_generator.data_format))
    print('dir: {0}'.format(validation_generator.directory))
    print('image_shape: {0}'.format(validation_generator.image_shape))
    print('target_size: {0}'.format(validation_generator.target_size))
    print('total_batches_seen: {0}'.format(validation_generator.total_batches_seen))

    print('\n')

    x_batch, y_batch = next(train_generator)
    for i in range (0, 32):
        img = x_batch[i]
        label = y_batch[i]
        print(label)

    """
    test_generator = ImageDataGenerator().flow_from_directory(
        test_dir,
        target_size=(resize_dim[0], resize_dim[1]),
        batch_size=batch_size,
        class_mode="categorical", # Not sure about this
        shuffle=False,
        save_to_dir=None)
    """

    # Take images from dir in batches
    model.fit_generator(train_generator,
                        steps_per_epoch=int(num_train_images/batch_size),
                        epochs=epochs)

# Note we can do the score part of ourselves (predicing labels by taking img one by one

# Score trained model on validation data
validation_scores = model.evaluate_generator(
                                  validation_generator,
                                  steps=int(num_validation_images/batch_size))

# Index is 0 because we have provided only one metric in model?! Not sure!
print('Validation Accuracy:', validation_scores[1])

"""
# Score trained model on test data
test_scores = model.evaluate_generator(self,
                                  test_generator,
                                  (num_test_images/batch_size),
                                  workers=4,
                                  use_multiprocessing=False)

# Index is 0 because we have provided only one metric in model?! Not sure!
print('Test Accuracy:', test_scores[0])
"""
