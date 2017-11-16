from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os

batch_size = 32
num_classes = 43
epochs = 50
data_augmentation = True
train_dir = "fill"
validation_dir = "fill"
test_dir= "fill"
num_train_images = "fill"
num_validation_images = "fill"
num_test_images = "fill"
resize_dim = "fill"

# Create CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
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
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

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
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 zca_epsilon=1e-6,
                                 rotation_range=0.,
                                 width_shift_range=0.,
                                 height_shift_range=0.,
                                 shear_range=0.,
                                 zoom_range=0.,
                                 channel_shift_range=0.,
                                 fill_mode='nearest',
                                 cval=0.,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 rescale=None,
                                 preprocessing_function=None,
                                 data_format=K.image_data_format())

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(resize_dim[0], resize_dim[1]), # Dimensions to which all the images will be resized
        batch_size=batch_size, # Default value
        class_mode='categorical', # Not sure about this
        shuffle=False, # Might help (causes problems if True,sometimes)
        save_to_dir=None) # Saving to visualize the augmented images

    validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(resize_dim[0], resize_dim[1]),
        batch_size=batch_size,
        class_mode="categorical", # Not sure about this
        shuffle=False,
        save_to_dir=None)

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(resize_dim[0], resize_dim[1]),
        batch_size=batch_size,
        class_mode="categorical", # Not sure about this
        shuffle=False,
        save_to_dir=None)

    print('Printing the train_data files')
    print train_generator.filenames

    # Take images from dir in batches
    model.fit_generator(train_generator,
                        steps_per_epoch=(num_train_images/batch_size),
                        epochs=epochs,
                        workers=4,
                        use_multiprocessing=False,
                        validation_data=validation_generator,
                        validation_steps=(num_validation_images/batch_size))

# Note we can do the score part of ourselves (predicing labels by taking img one by one

# Score trained model on validation data
validation_scores = model.evaluate_generator(self,
                                  validation_generator,
                                  (num_validation_images/batch_size),
                                  workers=4,
                                  use_multiprocessing=False)

# Index is 0 because we have provided only one metric in model?! Not sure!
print('Validation Accuracy:', validation_scores[0])

# Score trained model on test data
test_scores = model.evaluate_generator(self,
                                  test_generator,
                                  (num_test_images/batch_size),
                                  workers=4,
                                  use_multiprocessing=False)

# Index is 0 because we have provided only one metric in model?! Not sure!
print('Test Accuracy:', test_scores[0])
