import tensorflow as tf
from tensorflow.keras import layers, models


def train_cnn(model, train_ds, val_ds, epochs, lr, save):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    csv_logger = tf.keras.callbacks.CSVLogger('epochs.csv')

    history = model.fit(
        train_ds,
        epochs=epochs, 
        validation_data = val_ds, 
        callbacks = [csv_logger]
    )
    if save:
        model_name = "asl_class_" + str(epochs) + str(lr) + ".keras"
        model.save(model_name)

    return history