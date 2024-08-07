import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import pathlib
import PIL
import PIL.Image
import pprint
from plotnine import *

from train_cnn import train_cnn
from create_cnn import create_cnn
from show_imgs import show_imgs
from classify_single_img import classify_single_img
from handdetection import liveDetect

# This is the folder that contains our training data
directory = "/Users/jasonkuchtey/Desktop/asl_data/archive/asl_alphabet_train/asl_alphabet_train"
# This random seed makes our results recreateable
seed = 12345
# Determines the percentage of data dedicated to training and validation
split = 0.7
# How many epochs our model will trainfor
epochs = 5
# Determines how often our weights are adjusted
lr = 0.001
# If we want to check a single image, input the filename here
image = "f1.jpg"
# What size our images will be adjusted to
image_size = 100
# If true, our model will be saved as a .keras model with the learning rate and epoch count in the title
save = False
# If we want to use a saved model, put filename here
saved_model = "asl_class_50.001.keras"


train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory, 
    shuffle=True, 
    seed=seed, 
    validation_split=split, 
    subset="both", 
    label_mode="categorical", 
    labels="inferred", 
    image_size=(image_size, image_size), 
    batch_size=32
)

class_names = train_ds.class_names
# show_imgs(train_ds, class_names)
# ONLY WORKS IF YOU REMOVE labels AND label_mode



# Todo: add image size as create_cnn function param
if save:
    ourCNN = create_cnn(image_size)
    history = train_cnn(ourCNN, train_ds, val_ds, epochs, lr, save)
    test_loss, test_acc = ourCNN.evaluate(val_ds, verbose=1)
else:
    model = tf.keras.models.load_model(saved_model)
    # test_loss, test_acc = model.evaluate(val_ds, verbose=1)
    # classify_single_img(image, image_size, model, class_names)


# liveDetect(predict=True, class_names=class_names, image_size=image_size)

# classify_single_img(image, image_size, ourCNN, class_names)