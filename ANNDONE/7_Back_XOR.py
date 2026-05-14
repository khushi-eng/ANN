"""Write a python program to show Back Propagation Network for XOR function with Binary
Input and Output"""
import numpy as np


# ---------------- INPUT DATASET ----------------

# XOR inputs

X = np.array([

    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]

])



# ---------------- TARGET OUTPUTS ----------------

# XOR outputs

y = np.array([

    [0],
    [1],
    [1],
    [0]

])



# ---------------- SIGMOID ACTIVATION FUNCTION ----------------

# Converts values between 0 and 1

def sigmoid(x):

    # np.exp() calculates exponential value

    return 1 / (1 + np.exp(-x))



# ---------------- SIGMOID DERIVATIVE ----------------

# Used in backpropagation

def sigmoid_derivative(x):

    return x * (1 - x)



# ---------------- INITIALIZE WEIGHTS ----------------

# np.random.rand() generates random values


# Weights between input layer and hidden layer

# 2 input neurons → 2 hidden neurons

W1 = np.random.rand(2, 2)



# Weights between hidden layer and output layer

# 2 hidden neurons → 1 output neuron

W2 = np.random.rand(2, 1)



# ---------------- INITIALIZE BIASES ----------------

# Bias for hidden layer

b1 = np.random.rand(1, 2)


# Bias for output layer

b2 = np.random.rand(1, 1)



# ---------------- LEARNING RATE ----------------

learning_rate = 0.1



# ---------------- TRAINING ----------------

# Number of iterations

epochs = 10000



# Training loop

for i in range(epochs):


    # ---------------- FORWARD PROPAGATION ----------------

    # Input layer → Hidden layer

    # np.dot() performs matrix multiplication

    Z1 = np.dot(X, W1) + b1


    # Apply sigmoid activation

    A1 = sigmoid(Z1)



    # Hidden layer → Output layer

    Z2 = np.dot(A1, W2) + b2


    # Final output

    output = sigmoid(Z2)



    # ---------------- ERROR CALCULATION ----------------

    # Difference between actual and predicted output

    error = y - output



    # ---------------- BACKPROPAGATION ----------------

    # Calculate output layer delta

    d_output = error * sigmoid_derivative(output)



    # Calculate hidden layer error

    # W2.T means transpose of W2

    error_hidden = d_output.dot(W2.T)



    # Calculate hidden layer delta

    d_hidden = error_hidden * sigmoid_derivative(A1)



    # ---------------- UPDATE WEIGHTS ----------------

    # A1.T means transpose of hidden layer output

    W2 += A1.T.dot(d_output) * learning_rate



    # X.T means transpose of input matrix

    W1 += X.T.dot(d_hidden) * learning_rate



    # ---------------- UPDATE BIASES ----------------

    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate



# ---------------- DISPLAY OUTPUT ----------------

print("Final Outputs after Training:\n")

print(output)



# ---------------- PREDICTIONS ----------------

print("\nPredicted XOR Outputs:\n")


for i in range(len(X)):


    # If output ≥ 0.5 → 1
    # Else → 0

    prediction = 1 if output[i] >= 0.5 else 0


    print(X[i], "->", prediction)


"""# Dry Run of Backpropagation Network for XOR Function

---

# STEP 1: Input Dataset

```python id="1"
X = np.array([
 [0,0],
 [0,1],
 [1,0],
 [1,1]
])
```

These are XOR inputs.

| Input 1 | Input 2 |
| ------- | ------- |
| 0       | 0       |
| 0       | 1       |
| 1       | 0       |
| 1       | 1       |

---

# STEP 2: Target Outputs

```python id="2"
y = np.array([
 [0],
 [1],
 [1],
 [0]
])
```

XOR truth table:

| Input   | Output |
| ------- | ------ |
| 0 XOR 0 | 0      |
| 0 XOR 1 | 1      |
| 1 XOR 0 | 1      |
| 1 XOR 1 | 0      |

---

# STEP 3: Initialize Weights

Suppose randomly generated weights are:

---

## Input → Hidden Weights

```python id="3"
W1 =
[
 [0.2, 0.4],
 [0.3, 0.1]
]
```

Shape:

```python id="4"
(2,2)
```

Meaning:

* 2 input neurons
* 2 hidden neurons

---

## Hidden → Output Weights

```python id="5"
W2 =
[
 [0.5],
 [0.7]
]
```

Shape:

```python id="6"
(2,1)
```

---

# STEP 4: Initialize Biases

Suppose:

```python id="7"
b1 = [[0.1,0.2]]
```

Hidden layer bias.

---

```python id="8"
b2 = [[0.3]]
```

Output layer bias.

---

# STEP 5: Forward Propagation

We take first input:

```python id="9"
[0,0]
```

---

# Input → Hidden Layer

```python id="10"
Z1 = np.dot(X,W1) + b1
```

Formula:

Z_1 = XW_1 + b_1

For input `[0,0]`:

---

## Hidden Neuron 1

[
(0\times0.2)+(0\times0.3)+0.1
]

[
=0.1
]

---

## Hidden Neuron 2

[
(0\times0.4)+(0\times0.1)+0.2
]

[
=0.2
]

So:

```python id="11"
Z1 = [0.1,0.2]
```

---

# Apply Sigmoid Activation

```python id="12"
A1 = sigmoid(Z1)
```

Formula:

\sigma(x)=\frac{1}{1+e^{-x}}

---

## For 0.1

[
\sigma(0.1)=0.524
]

---

## For 0.2

[
\sigma(0.2)=0.549
]

So:

```python id="13"
A1 = [0.524,0.549]
```

---

# Hidden → Output Layer

```python id="14"
Z2 = np.dot(A1,W2) + b2
```

Formula:

Z_2 = A_1W_2 + b_2

Calculation:

[
(0.524\times0.5)+(0.549\times0.7)+0.3
]

[
=0.262+0.384+0.3
]

[
=0.946
]

So:

```python id="15"
Z2 = 0.946
```

---

# Final Output

```python id="16"
output = sigmoid(Z2)
```

[
\sigma(0.946)=0.720
]

So:

```python id="17"
output = 0.720
```

---

# STEP 6: Error Calculation

Actual target for `[0,0]`:

```python id="18"
0
```

Predicted:

```python id="19"
0.720
```

---

```python id="20"
error = y - output
```

[
0 - 0.720
]

[
=-0.720
]

So:

```python id="21"
error = -0.720
```

---

# STEP 7: Output Delta

```python id="22"
d_output = error * sigmoid_derivative(output)
```

Sigmoid derivative:

\sigma'(x)=x(1-x)

For output 0.720:

[
0.720(1-0.720)
]

[
=0.2016
]

Multiply with error:

[
-0.720 \times 0.2016
]

[
=-0.145
]

So:

```python id="23"
d_output = -0.145
```

---

# STEP 8: Hidden Layer Error

```python id="24"
error_hidden = d_output.dot(W2.T)
```

Transpose:

```python id="25"
W2.T = [0.5,0.7]
```

Calculation:

[
-0.145 \times [0.5,0.7]
]

[
=[-0.0725,-0.1015]
]

So:

```python id="26"
error_hidden =
[-0.0725,-0.1015]
```

---

# STEP 9: Hidden Delta

```python id="27"
d_hidden = error_hidden * sigmoid_derivative(A1)
```

For hidden neuron 1:

[
0.524(1-0.524)
]

[
=0.249
]

Multiply:

[
-0.0725 \times 0.249
]

[
=-0.018
]

Similarly second neuron:

[
-0.1015 \times 0.247
]

[
=-0.025
]

So:

```python id="28"
d_hidden =
[-0.018,-0.025]
```

---

# STEP 10: Update W2

```python id="29"
W2 += A1.T.dot(d_output) * learning_rate
```

Learning rate:

```python id="30"
0.1
```

Weight correction:

[
0.524 \times (-0.145)\times0.1
]

[
=-0.0075
]

Update:

Old weight:

```python id="31"
0.5
```

New weight:

```python id="32"
0.4925
```

Similarly all weights update.

---

# STEP 11: Update W1

```python id="33"
W1 += X.T.dot(d_hidden) * learning_rate
```

Weights between input-hidden layers improve.

---

# STEP 12: Update Biases

```python id="34"
b2 += np.sum(d_output) * learning_rate
```

New output bias:

[
0.3 + (-0.145\times0.1)
]

[
=0.2855
]

---

```python id="35"
b1 += np.sum(d_hidden) * learning_rate
```

Hidden biases also update.

---

# STEP 13: Repeat 10000 Times

```python id="36"
for i in range(epochs):
```

Network repeatedly:

* predicts
* calculates error
* updates weights

Until error becomes very small.

---

# STEP 14: Final Outputs After Training

Suppose network learns:

```python id="37"
[
 [0.02],
 [0.97],
 [0.96],
 [0.03]
]
```

---

# STEP 15: Predictions

```python id="38"
prediction = 1 if output[i] >= 0.5 else 0
```

Rules:

* ≥ 0.5 → 1
* < 0.5 → 0

---

# Final Prediction Table

| Input | Network Output | Prediction |
| ----- | -------------- | ---------- |
| [0,0] | 0.02           | 0          |
| [0,1] | 0.97           | 1          |
| [1,0] | 0.96           | 1          |
| [1,1] | 0.03           | 0          |

---

# Final Output

```python id="39"
[0 0] -> 0
[0 1] -> 1
[1 0] -> 1
[1 1] -> 0
```

The neural network successfully learns XOR logic using backpropagation.
"""