from __future__ import print_function
import keras                #Keras for deeplearnig......
from keras.preprocessing.image import ImageDataGenerator    #for generating images from an images...
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

num_classes = 5                 #this is for how  many clases I have imported or using..
img_rows, img_cols = 48, 48     # this is total 48*48 imeges  to train.....
batch_size = 8                #how many images I want to give my model to train at once time......

# most of us make batch of 32 but due to gpu and memory i take it 8....

train_data_dir = 'F:/ML Project/Facial-Expressions-Recognition1/images/train'
validation_data_dir = 'F:/ML Project/Facial-Expressions-Recognition1/images/validation'

#Now its time to generate more images from one image..

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, shear_range=0.3, zoom_range=0.3, width_shift_range=0.4,
                                   height_shift_range=0.4, horizontal_flip=True, fill_mode='nearest')

#valideting generated image in training phase.......
validation_datagen = ImageDataGenerator(rescale=1./255)


#this is for trainign model input data.......
train_generator = train_datagen.flow_from_directory(train_data_dir, color_mode='grayscale', target_size=(img_rows, img_cols),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True)

#and target === output........vala scene...
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, color_mode='grayscale',
                                                               target_size=(img_rows, img_cols), batch_size=batch_size,
                                                               class_mode='categorical', shuffle=True)
