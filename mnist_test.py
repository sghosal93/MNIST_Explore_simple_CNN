#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:26:11 2019

@author: samghosal
"""
from __future__ import division

"""----------------------------------------------------------------------------------------------   
   README: Simple Python Code for Testing and evaluating the trained CNN model on MNIST test data
   ----------------------------------------------------------------------------------------------   
   For each section of the code, there are Headings depicting what the subsequent lines of code
   do. 
   
   Evaluation Metrics: Test Loss, Test  Accuracy, Confusion Matrix, Precision, Recall and F1-Score 
   Test Accuracy achieved: 99.23%. 
   PLease check README within the training code - 'mnist_train.py' for other details.
""" 

###################
# IMPORT PACKAGES #
###################

import numpy as np
import gzip
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras.utils import to_categorical
# from keras.layers.normalization import BatchNormalization
from keras import backend as K
from model import load_model

K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto(device_count = {'GPU': 1})
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
 
np.random.seed(1337) # set seed for reproducability

###########################################
# LOADING THE TEST DATA and PREPROCESSING #
###########################################

# write files to directory
filename3 = "MNIST/t10k-images-idx3-ubyte.gz"
filename4 = "MNIST/t10k-labels-idx1-ubyte.gz"

# image parameters
img_size = 784 # size (vectorized sample)

# input image dimensions
img_rows, img_cols = 28, 28 # information obtained from metadata
    
with gzip.open(filename3, 'rb') as f:
    test_img = np.frombuffer(f.read(), np.uint8, offset=16)
    test_img = test_img.reshape(-1, img_size)
    f.close()
    
with gzip.open(filename4, 'rb') as f:
    test_lab = np.frombuffer(f.read(), np.uint8, offset=8)
    f.close()

X_test = test_img
Y_test_pre = test_lab

num_classes = 10

# Transform to Categorical Variables (One-Hot Encode)
Y_test = to_categorical(Y_test_pre, num_classes=num_classes)

# Reshape and Normalize the Image data
if K.image_data_format() == 'channels_first':
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Normalize the data and cast to float32
X_test = X_test/255
X_test = X_test.astype('float32')

print('Test Data shape...................')
print('Test data shape:', X_test.shape)

##########################
# Load Model and weights #
##########################
    
model = load_model(num_classes, True)

#####################################
# Evaluating loaded model & weights #
#####################################

# Test Data Scores
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Getting Test Predictions
y_test_predictions = model.predict(x=X_test)
y_test_class = np.argmax(Y_test, axis=1)
y_pred_class = np.argmax(y_test_predictions, axis=1)

# Evaluation Metrics
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# Print Classification Report
print('Printing Classification Report (per class Precision, Recall and F1-Score): ')
print(classification_report(Y_test_pre, y_pred_class))

# Plot Confusion Matrix
cm = confusion_matrix(Y_test_pre, y_pred_class)
cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

print('Printing Confusion Matrix: ')
print(cm)

plt.figure()
plt.imshow(cm, cmap = 'plasma')
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, np.arange(num_classes), rotation=45)
plt.yticks(tick_marks, np.arange(num_classes))

fmt = '.1f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), fontsize=6,
             horizontalalignment="center",
             color="red" if cm[i, j] > thresh else "white")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('results/cm.png')
