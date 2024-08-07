import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def classify_single_img(image, image_size, model, class_names, npimage=False):
    if not npimage:
        img = tf.keras.utils.load_img(image, target_size=(image_size, image_size))
    else:
        img = tf.keras.utils.array_to_img(image, scale=True)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    img_predictions = model.predict(img_array)
    print(img_predictions)

    pred_label = class_names[np.argmax(np.round(img_predictions,2))]
    print(" Predicted label is :: "+ pred_label)

    #if you also want to display the image that was passed use the code below
    plt.imshow(img)


# DOESN'T WORK YET