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

path="/home/christian/Desktop/UNIV/DEMETER/TRAINING/PLANT-CONCRETE-PREP/"
input_shape = cv2.imread("/home/christian/Desktop/UNIV/DEMETER/TRAINING/PLANT-CONCRETE-PREP/40/0.png").shape[0:2]
print("image shape: "+str(input_shape))

#create dataset
batch_size = 128

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=path,
    label_mode='categorical',
    batch_size=batch_size,
    color_mode='grayscale',
    image_size=input_shape,
    validation_split=0.2,
    subset="both",
    seed=42,
)
# Print the class names and number of classes
class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names)
print("Number of classes:", num_classes)

input_shape=input_shape+(1,)
print("input shape: "+str(input_shape))

print("Dataset data :")
print(train_ds.element_spec)

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
def make_testmodel(input_shape, num_classes):
    x = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return x

model = make_testmodel(input_shape=input_shape, num_classes=num_classes)
keras.utils.plot_model(model, show_shapes=True)

epochs = 25

callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]
model.compile(optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],)
model.fit(train_ds,epochs=epochs,batch_size=50,callbacks=callbacks,validation_data=val_ds)