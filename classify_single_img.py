import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


def classify_single_img(image_size, model, class_names):
    img = keras.utils.load_img('c1.jpg', target_size=image_size)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    img_predictions = model.predict(img_array)

    pred_label = class_names[np.argmax(np.round(img_predictions,2))]
    print(" Predicted label is :: "+ pred_label)

    #if you also want to display the image that was passed use the code below
    plt.imshow(img)