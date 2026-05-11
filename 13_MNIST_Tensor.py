# MNIST Handwritten Digit Detection using TensorFlow

# TensorFlow library for deep learning / neural networks
import tensorflow as tf

# mnist dataset contains handwritten digit images (0–9)
from tensorflow.keras.datasets import mnist

# Sequential() creates neural network layer-by-layer
from tensorflow.keras.models import Sequential

# Dense = fully connected neural network layer
# Flatten = converts 2D image into 1D vector
from tensorflow.keras.layers import Dense, Flatten

# Adam optimizer updates weights during training
from tensorflow.keras.optimizers import Adam

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
# each image size = 28 × 28 pixels
# labels = digits from 0 to 9


# ---------------- NORMALIZE DATA ----------------

# Pixel values originally range from 0 to 255

# Dividing by 255 converts values into range 0–1

# Smaller values help network learn faster

X_train = X_train / 255.0
X_test = X_test / 255.0


# ---------------- CREATE MODEL ----------------

# Sequential() builds neural network layer-by-layer

model = Sequential([


    # ---------------- FLATTEN LAYER ----------------

    # Flatten converts 28×28 image into 784-element vector

    # input_shape=(28,28)
    # means input image size is 28×28

    Flatten(input_shape=(28, 28)),


    # ---------------- HIDDEN LAYER ----------------

    # Dense means fully connected layer

    # 128 neurons used in hidden layer

    # activation='relu'
    # ReLU removes negative values

    Dense(

        128,
        activation='relu'

    ),


    # ---------------- OUTPUT LAYER ----------------

    # 10 neurons because digits are 0–9

    # softmax converts outputs into probabilities

    Dense(

        10,
        activation='softmax'

    )

])


# ---------------- COMPILE MODEL ----------------

# compile() configures training process

model.compile(

    # Adam optimizer with learning rate
    optimizer=Adam(learning_rate=0.001),

    # loss function for multi-class classification
    loss='sparse_categorical_crossentropy',

    # accuracy used to measure performance
    metrics=['accuracy']

)


# ---------------- TRAIN MODEL ----------------

# fit() trains neural network

# X_train → input images
# y_train → correct digit labels

# batch_size=64
# network trains using 64 images at one time

# epochs=5
# complete dataset trained 5 times

model.fit(

    X_train,
    y_train,
    batch_size=64,
    epochs=5

)


# ---------------- EVALUATE MODEL ----------------

# evaluate() checks performance on test data

# returns:
# loss value
# accuracy value

loss, accuracy = model.evaluate(

    X_test,
    y_test

)


# ---------------- DISPLAY RESULTS ----------------

print("\nTest Loss:", loss)

print("Test Accuracy:", accuracy)


# ---------------- PREDICT TEST IMAGES ----------------

# predict() generates probability outputs

# X_test[:5]
# first 5 test images

predictions = model.predict(X_test[:5])


# ---------------- DISPLAY PREDICTED DIGITS ----------------

print("\nPredicted Digits:")


# np.argmax()
# returns index of largest probability

# axis=1 means row-wise maximum

print(np.argmax(predictions, axis=1))


# ---------------- DISPLAY ACTUAL DIGITS ----------------

print("\nActual Digits:")

print(y_test[:5])


"""# Dry Run of MNIST Handwritten Digit Detection Program

---

# Step 1: Load Dataset

```python id="1"
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

MNIST dataset contains handwritten digit images.

Each image size:

```python id="2"
28 × 28
```

Example:

```python id="3"
X_train.shape
```

Output:

```python id="4"
(60000,28,28)
```

Meaning:

* 60000 training images
* each image = 28×28 pixels

---

# Step 2: Normalize Data

```python id="5"
X_train = X_train / 255.0
```

Pixel values originally:

```python id="6"
0 → 255
```

Example:

Original pixel:

```python id="7"
255
```

After normalization:

[
255/255 = 1
]

Original pixel:

```python id="8"
128
```

After normalization:

[
128/255 = 0.50
]

Now all values lie between:

```python id="9"
0 and 1
```

---

# Step 3: Create Neural Network

```python id="10"
model = Sequential([...])
```

Network layers:

```text id="11"
Input → Hidden Layer → Output Layer
```

---

# Step 4: Flatten Layer

```python id="12"
Flatten(input_shape=(28,28))
```

Converts image into 1D vector.

28×28 image:

```text id="13"
784 pixels
```

So:

```python id="14"
28 × 28 = 784
```

Image becomes:

```python id="15"
[0.0,0.5,1.0,0.2,....]
```

---

# Step 5: Hidden Layer

```python id="16"
Dense(128,activation='relu')
```

Meaning:

* hidden layer has 128 neurons

Each neuron calculates:

Z = WX + b

Then ReLU activation applied:

ReLU(x)=\max(0,x)

Example:

| Input | ReLU Output |
| ----- | ----------- |
| -3    | 0           |
| 5     | 5           |

---

# Step 6: Output Layer

```python id="17"
Dense(10,activation='softmax')
```

10 neurons because:

* digits = 0 to 9

Softmax converts outputs into probabilities.

Example:

```python id="18"
[0.01,0.02,0.90,0.01,...]
```

Highest probability = predicted digit.

---

# Step 7: Compile Model

```python id="19"
model.compile(...)
```

Uses:

* Adam optimizer
* sparse_categorical_crossentropy loss
* accuracy metric

---

# Step 8: Train Model

```python id="20"
model.fit(X_train,y_train,batch_size=64,epochs=5)
```

Training starts.

---

## First Batch

64 images given together.

Suppose first image is digit:

```python id="21"
5
```

Network predicts:

```python id="22"
[0.1,0.05,0.02,0.01,0.03,0.60,...]
```

Highest probability:

```python id="23"
0.60
```

at index:

```python id="24"
5
```

Prediction correct.

---

## Another Example

Actual digit:

```python id="25"
7
```

Prediction:

```python id="26"
[0.2,0.1,0.05,0.04,0.01,0.03,0.02,0.30,...]
```

Prediction weak.

Error calculated.

Weights updated using backpropagation.

---

# Step 9: Repeat for 5 Epochs

```python id="27"
epochs=5
```

Entire dataset trained 5 times.

Network gradually improves.

---

# Step 10: Evaluate Model

```python id="28"
loss,accuracy = model.evaluate(X_test,y_test)
```

Suppose output:

```python id="29"
Accuracy = 0.97
```

Meaning:

```text id="30"
97% images predicted correctly
```

---

# Step 11: Predict First 5 Test Images

```python id="31"
predictions = model.predict(X_test[:5])
```

Suppose probabilities:

```python id="32"
[
 [0.01,0.01,0.95,...],
 [0.90,0.02,0.01,...],
 ...
]
```

---

# Step 12: Find Largest Probability

```python id="33"
np.argmax(predictions,axis=1)
```

Suppose result:

```python id="34"
[2,0,1,4,9]
```

These are predicted digits.

---

# Step 13: Compare Actual Digits

```python id="35"
y_test[:5]
```

Suppose:

```python id="36"
[2,0,1,4,9]
```

Predictions match actual values.

---

# Final Output

```python id="37"
Predicted Digits:
[2 0 1 4 9]

Actual Digits:
[2 0 1 4 9]
```

Neural network successfully recognizes handwritten digits.
"""