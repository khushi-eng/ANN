# Assignment Title:
# Train a Neural Network with TensorFlow and
# Evaluation of Logistic Regression using TensorFlow

# TensorFlow library for deep learning / neural network operations
import tensorflow as tf

# NumPy library for array and mathematical operations
import numpy as np

# train_test_split() divides dataset into training and testing data
from sklearn.model_selection import train_test_split

# StandardScaler() standardizes/scales the dataset
from sklearn.preprocessing import StandardScaler

# load_breast_cancer() loads breast cancer dataset
from sklearn.datasets import load_breast_cancer


# ---------------- LOAD DATASET ----------------

# Load breast cancer dataset
data = load_breast_cancer()


# ---------------- INPUT AND OUTPUT ----------------

# data.data contains all input features
# Example:
# radius, texture, area, perimeter etc.
X = data.data


# data.target contains output labels
# 0 = malignant
# 1 = benign
y = data.target


# ---------------- SPLIT DATA ----------------

# train_test_split() divides data into:
# training data → used for learning
# testing data → used for evaluation

# test_size=0.20
# 20% data used for testing

# random_state=42
# ensures same random split every time

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42

)


# ---------------- SCALE DATA ----------------

# StandardScaler() standardizes data
# Formula:
# (x - mean) / standard deviation

sc = StandardScaler()


# fit_transform()
# calculates mean/std and transforms training data

X_train = sc.fit_transform(X_train)


# transform()
# scales test data using same training parameters

X_test = sc.transform(X_test)


# ---------------- CREATE MODEL ----------------

# tf.keras.models.Sequential()
# creates neural network layer-by-layer

model = tf.keras.models.Sequential([

    # Dense layer means fully connected layer

    tf.keras.layers.Dense(

        # 1 output neuron
        1,

        # sigmoid activation gives output between 0 and 1
        activation='sigmoid',

        # input_shape specifies number of input features
        input_shape=(X_train.shape[1],)

    )

])


# ---------------- COMPILE MODEL ----------------

# compile() configures model for training

model.compile(

    # adam optimizer updates weights automatically
    optimizer='adam',

    # binary_crossentropy used for binary classification
    loss='binary_crossentropy',

    # accuracy metric used to measure performance
    metrics=['accuracy']

)


# ---------------- TRAIN MODEL ----------------

# fit() trains neural network

# X_train → input training data
# y_train → correct output labels

# epochs=5
# complete dataset trained 5 times

model.fit(

    X_train,
    y_train,
    epochs=5

)


# ---------------- PREDICT OUTPUT ----------------

# predict() generates output predictions for test data

y_pred = model.predict(X_test)


# ---------------- EVALUATE MODEL ----------------

# evaluate() calculates:
# loss value
# accuracy value

test_loss, test_accuracy = model.evaluate(

    X_test,
    y_test

)


# ---------------- DISPLAY RESULT ----------------

print("Accuracy is:", test_accuracy)

"""# Dry Run of StandardScaler

Suppose:

```python id="1"
X_train =
[
 [10],
 [20],
 [30]
]
```

---

# Step 1: Create Scaler

```python id="2"
sc = StandardScaler()
```

Scaler object is created.

---

# Step 2: fit_transform()

```python id="3"
X_train = sc.fit_transform(X_train)
```

This performs:

1. fit()
2. transform()

---

# Step 3: Calculate Mean

Formula:

\mu = \frac{\sum x}{n}

[
(10+20+30)/3
]

[
=60/3
]

[
=20
]

Mean = 20

---

# Step 4: Calculate Standard Deviation

Formula:

\sigma = \sqrt{\frac{\sum (x-\mu)^2}{n}}

[
(10-20)^2 = 100
]

[
(20-20)^2 = 0
]

[
(30-20)^2 = 100
]

Sum:

[
100+0+100=200
]

Divide by 3:

[
200/3=66.67
]

Square root:

[
\sqrt{66.67}=8.16
]

Standard deviation ≈ 8.16

---

# Step 5: Apply Scaling Formula

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"z = \frac{x-\mu}{\sigma}"} }

---

## For 10

[
(10-20)/8.16
]

[
=-10/8.16
]

[
=-1.22
]

---

## For 20

[
(20-20)/8.16
]

[
=0
]

---

## For 30

[
(30-20)/8.16
]

[
=10/8.16
]

[
=1.22
]

---

# Final Scaled Training Data

```python id="4"
X_train =
[
 [-1.22],
 [ 0   ],
 [ 1.22]
]
```

---

# Step 6: Test Data

Suppose:

```python id="5"
X_test =
[
 [40]
]
```

---

# Step 7: Transform Test Data

```python id="6"
X_test = sc.transform(X_test)
```

Uses SAME:

* mean = 20
* std = 8.16

---

Calculation:

[
(40-20)/8.16
]

[
=20/8.16
]

[
=2.45
]

---

# Final Scaled Test Data

```python id="7"
X_test =
[
 [2.45]
]
```

---

# Final Result

| Original | Scaled |
| -------- | ------ |
| 10       | -1.22  |
| 20       | 0      |
| 30       | 1.22   |
| 40       | 2.45   |

StandardScaler converts data into similar range for better neural network learning.
"""