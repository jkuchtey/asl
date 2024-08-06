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

directory = "/Users/jasonkuchtey/Downloads/archive/asl_alphabet_train/asl_alphabet_train"
seed = 12345
split = 0.7
epochs = 5
lr = 0.001
image_size = (180, 180)

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory, 
    shuffle=True, 
    seed=seed, 
    validation_split=split, 
    subset="both", 
    label_mode="categorical", 
    labels="inferred", 
    image_size=(32, 32), 
    batch_size=32
)

class_names = train_ds.class_names
# show_imgs(train_ds, class_names)
# ONLY WORKS IF YOU REMOVE labels AND label_mode




# Todo: add image size as create_cnn function param
ourCNN = create_cnn()
history = train_cnn(ourCNN, train_ds, val_ds, epochs, lr)
test_loss, test_acc = ourCNN.evaluate(val_ds, verbose=1)
# classify_single_img(image_size, ourCNN, class_names)

