import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn():
    model = models.Sequential()
    tf.keras.layers.Rescaling(1./255)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    #ouput layer: 29 is the numer of labels
    model.add(layers.Dense(29, activation=None))


    return model