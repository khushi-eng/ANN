# CNN Implementation using TensorFlow

# TensorFlow library for deep learning / neural networks
import tensorflow as tf

# mnist dataset contains handwritten digit images (0–9)
from tensorflow.keras.datasets import mnist

# Sequential() creates neural network layer-by-layer
from tensorflow.keras.models import Sequential

# Conv2D → convolution layer
# MaxPooling2D → pooling layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Flatten converts 2D data into 1D
# Dense means fully connected layer
from tensorflow.keras.layers import Flatten, Dense

# NumPy library for mathematical operations
import numpy as np


# ---------------- LOAD MNIST DATASET ----------------

# mnist.load_data() loads:
# training images + labels
# testing images + labels

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# X_train → training images
# y_train → training labels

# X_test → testing images
# y_test → testing labels


# Example:
# image = 28 × 28 pixels
# label = digit value (0–9)


# ---------------- NORMALIZE DATA ----------------

# Pixel values are originally between 0 and 255

# Dividing by 255 converts values into range 0–1

# Smaller values help neural network learn faster

X_train = X_train / 255.0
X_test = X_test / 255.0


# ---------------- RESHAPE DATA ----------------

# CNN expects 4D input:
# (samples, height, width, channels)

# -1 means TensorFlow automatically calculates number of images

# 28 × 28 → image size

# 1 means grayscale image
# (only one color channel)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# ---------------- CREATE CNN MODEL ----------------

# Sequential() builds model layer-by-layer

model = Sequential([


    # ---------------- CONVOLUTION LAYER ----------------

    # Conv2D creates feature maps from images

    # 32 = number of filters

    # (3,3) = filter/kernel size

    # activation='relu'
    # ReLU removes negative values

    # input_shape = image size

    Conv2D(

        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)

    ),


    # ---------------- MAX POOLING LAYER ----------------

    # Reduces image size

    # Keeps important features

    # (2,2) means 2×2 pooling window

    MaxPooling2D((2,2)),


    # ---------------- FLATTEN LAYER ----------------

    # Converts 2D feature maps into 1D vector

    Flatten(),


    # ---------------- OUTPUT LAYER ----------------

    # Dense = fully connected layer

    # 10 neurons for digits 0–9

    # softmax converts outputs into probabilities

    Dense(

        10,
        activation='softmax'

    )

])


# ---------------- COMPILE MODEL ----------------

# compile() configures training settings

model.compile(

    # adam optimizer updates weights automatically
    optimizer='adam',

    # sparse_categorical_crossentropy
    # used for multi-class classification

    loss='sparse_categorical_crossentropy',

    # accuracy used for performance measurement
    metrics=['accuracy']

)


# ---------------- TRAIN MODEL ----------------

# fit() trains neural network

# X_train → training images
# y_train → correct digit labels

# epochs=5
# dataset trained 5 times

model.fit(

    X_train,
    y_train,
    epochs=5

)


# ---------------- EVALUATE MODEL ----------------

# evaluate() checks model performance on test data

# returns:
# loss value
# accuracy value

loss, accuracy = model.evaluate(

    X_test,
    y_test

)


# ---------------- DISPLAY ACCURACY ----------------

print("Accuracy:", accuracy)


# ---------------- PREDICT TEST IMAGES ----------------

# predict() generates probability outputs

# X_test[:5]
# first 5 testing images

predictions = model.predict(X_test[:5])


# ---------------- DISPLAY PREDICTED VALUES ----------------

print("\nPredicted Values:")


# np.argmax()
# returns index of largest probability

# axis=1 means row-wise maximum

print(np.argmax(predictions, axis=1))


# ---------------- DISPLAY ACTUAL VALUES ----------------

print("\nActual Values:")

print(y_test[:5])