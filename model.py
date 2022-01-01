#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:26:11 2019

@author: samghosal
"""

"""-----------------------------------------------------------------------------------   
   README: Simple Python Code for Designing a CNN model for training on the MNIST data
   -----------------------------------------------------------------------------------   
   A 10-Class Classification Problem for MNIST: Model consists of an input convolutional layer, 
   followed by a second convolutional layer and then a max-pooling layer. The ouput of the pooling
   layer is then flattened after a 25% Dropout. This is followed by a fully-connected layer and 
   then finally the 10-unit softmax-classification layer after a 50% Dropout layer.
   
   A few models were tried out. Here, one such model is presented where a > 99% test accuracy is
   achieved at epoch 34. Previous model with lesser number of filters per convolutional layer (a 
   shallower model with lesser number of learning parameters) achieved similar accuracy but at 
   epoch 53, which is expected. The learning rate was kept at 0.0001 as a first step to prevent 
   overfitting and the network was kept shallow. The ReLU activation function was used where 
   necesary to take care of the evanishing/exploding gradients problem, which is typical of Deep
   Neural Networks. Dropout layers were also added to further take care of the "over-fitting"
   problem.
""" 

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

def load_model(num_classes,load_weight=False):
    
    num_classes = num_classes

    model = Sequential() 
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax')) # softmax classification layer
    
    opt = Adam(lr=0.0001) # custom learning rate
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    
    model.summary()

    if (load_weight==True):
        model.load_weights("MNIST_weights/threshold_weights.34-0.99-0.04.hdf5") # load trained weights during testing
    return model