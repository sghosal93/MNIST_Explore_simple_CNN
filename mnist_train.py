#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:26:11 2019

@author: samghosal
"""
from __future__ import division

"""---------------------------------------------------------------   
   README: Simple Python Code for Training a CNN on the MNIST data
   ---------------------------------------------------------------   
   VERSION INFORNMATION:
       supported on Python 2.x.x and Python 3.x.x
       Suggest use of the latest Anaconda Distribution and use a environment to use Tensorflow GPU
       Keras version: 2.2.4
       Tensorflow version: 1.10.0, 1.12.0 (conda environment)
       GPU used: NVIDIA GeForce GTX 1080 Ti (11176 MB)
   For each section of the code, there are Headings depicting what the subsequent lines of code
   do: 
       The first section imports required packages and allows for GPU use if it exists.
       The next section downloads and processes the data for training.
       The section right after defines a function to efficiently save a matplotlib figure
       The final section does a quick evaluation of the best trained model.
   Per epoch time: ~5s 
   Accuracy achieved: > 99.22% on MNIST Test Dataset
""" 

###################
# IMPORT PACKAGES #
###################

import numpy as np
import requests
import gzip
import os
import matplotlib.pyplot as plt
# from PIL import Image # import package for visualizing data sample
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras import backend as K
from model import load_model

# Configure Tensorflow environment
K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto(device_count = {'GPU': 1})
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
 
np.random.seed(1337) # set seed for reproducability

##########################################
# DOWNLOADING THE DATA and PREPROCESSING #
##########################################

# get download urls and send request
resp_train_img = requests.get("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", allow_redirects=True, stream=True)
resp_train_lab = requests.get("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", allow_redirects=True, stream=True)
resp_test_img = requests.get("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", allow_redirects=True, stream=True)
resp_test_lab = requests.get("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", allow_redirects=True, stream=True)

# # Required directories: If the directory does not exist, create it
if os.path.exists('MNIST') == False:
    os.mkdir('MNIST')

if os.path.exists('MNIST_weights') == False:
    os.mkdir('MNIST_weights')
    
if os.path.exists('results') == False:
    os.mkdir('results')

# write files to target directory
filename1 = "MNIST/train-images-idx3-ubyte.gz"
filename2 = "MNIST/train-labels-idx1-ubyte.gz"
filename3 = "MNIST/t10k-images-idx3-ubyte.gz"
filename4 = "MNIST/t10k-labels-idx1-ubyte.gz"

zfile = open(filename1, 'wb')
zfile.write(resp_train_img.content)
zfile.close()

zfile = open(filename2, 'wb')
zfile.write(resp_train_lab.content)
zfile.close()

zfile = open(filename3, 'wb')
zfile.write(resp_test_img.content)
zfile.close()

zfile = open(filename4, 'wb')
zfile.write(resp_test_lab.content)
zfile.close()

# image parameters
img_size = 784 # flattened size

# input image dimensions
img_rows, img_cols = 28, 28 # information obtained from metadata
    
with gzip.open(filename1, 'rb') as f:
    train_img = np.frombuffer(f.read(), np.uint8, offset=16)
    train_img = train_img.reshape(-1, img_size)
    f.close()
    os.remove(filename1) # comment this out if the file is to be kept
    
with gzip.open(filename2, 'rb') as f:
    train_lab = np.frombuffer(f.read(), np.uint8, offset=8)
    f.close()
    os.remove(filename2) # comment this out if the file is to be kept
    
with gzip.open(filename3, 'rb') as f:
    test_img = np.frombuffer(f.read(), np.uint8, offset=16)
    test_img = test_img.reshape(-1, img_size)
    f.close()
    # os.remove(filename3) # comment this out if the file is to be kept
    
with gzip.open(filename4, 'rb') as f:
    test_lab = np.frombuffer(f.read(), np.uint8, offset=8)
    f.close()
    # os.remove(filename4) # comment this out if the file is to be kept
'''   
# Check a random image from train/test dataset (SANITY CHECK)
def img_show(img): # function for displaying an image (uses the PIL package)
    # param: img - sample from dataset
    img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

I = train_img[0]
label = train_lab[0]
print(label)

img_show(I) # show the image
'''
# Generate training, validation and test data
X_train, X_valid, Y_train_pre, Y_valid_pre = train_test_split(train_img, train_lab, test_size = (1/6), random_state = 99)

X_test = test_img
Y_test_pre = test_lab

# One-Hot encode the labels
num_classes = 10

# Transform to Categorical Variables (One-Hot Encode)
Y_train = to_categorical(Y_train_pre, num_classes=num_classes)
Y_valid = to_categorical(Y_valid_pre, num_classes=num_classes)
Y_test = to_categorical(Y_test_pre, num_classes=num_classes)

# Reshape and Normalize the Image data
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
# Normalize the data and cast to float32
X_train = X_train/255
X_valid = X_valid/255
X_test = X_test/255

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')

print('Printing Train, Validation and Test Data shape..................................')
print('Training data shape:', X_train.shape)
print('Validation data shape:', X_valid.shape)
print('Test data shape:', X_test.shape)

##################################################
# Design and train a simple Sequential CNN Model #
##################################################
    
# Model Hyperparameters
batch_size = 128
epochs = 100

model = load_model(10, False)

callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=20, verbose=1),
            keras.callbacks.ModelCheckpoint(filepath='MNIST_weights/threshold_weights.{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5',
                                            monitor='val_acc', save_best_only=True, verbose=1),
        ]

# Train the Model
train_history = model.fit(X_train, Y_train,
	batch_size=batch_size, epochs=epochs, verbose=1, 
	validation_data=(X_valid, Y_valid), shuffle=True, callbacks=callbacks)

################################
# Function for saving a figure #
################################

def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")

##########################################
# Quick Evaluation for the trained model #
##########################################

# Get Score and Accurcy on Test Data
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Loss, Accuracy vs Epochs Plots

acc = train_history.history['acc']
val_acc = train_history.history['val_acc']

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val_loss'])
save("results/losses", ext="png", close=False, verbose=True)
plt.close()

plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc', 'val_acc'], loc=4)
save("results/accuracy", ext="png", close=False, verbose=True)
plt.close()

