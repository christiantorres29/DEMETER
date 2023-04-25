# Import lib's
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import layers
import cv2
import glob
import os

num_classes = len([40,60,70,80,90,100,110,120])
print(num_classes)

path="/home/christian/Desktop/UNIV/DEMETER/TRAINING/PLANT-CONCRETE-PREP/"
input_shape = cv2.imread("/home/christian/Desktop/UNIV/DEMETER/TRAINING/PLANT-CONCRETE-PREP/40/0.png").shape[0:2]

#create dataset
batch_size = 128

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=input_shape,
    batch_size=batch_size,
)

input_shape=input_shape+(1,)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=input_shape, num_classes=num_classes)
keras.utils.plot_model(model, show_shapes=True)

epochs = 25

callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]
model.compile(optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],)
model.fit(train_ds,epochs=epochs,callbacks=callbacks,validation_data=val_ds)