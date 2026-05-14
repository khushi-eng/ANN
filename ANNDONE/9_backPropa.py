# Back Propagation Feed-Forward Neural Network in Python

import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (XOR Problem)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected output
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Initialize weights randomly
np.random.seed(1)

input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Weights and bias initialization
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))

wo = np.random.uniform(size=(hidden_neurons, output_neurons))
bo = np.random.uniform(size=(1, output_neurons))

# Training parameters
learning_rate = 0.5
epochs = 10000

# Training using Backpropagation
for i in range(epochs):

    # Forward Propagation
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    predicted_output = sigmoid(final_input)

    # Error Calculation
    error = y - predicted_output

    # Back Propagation
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(wo.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Updating Weights and Biases
    wo += hidden_output.T.dot(d_output) * learning_rate
    bo += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Output after training
print("Predicted Output:")
print(predicted_output)

"""# Dry Run of Back Propagation Feed-Forward Neural Network

---

## Step 1: Input Dataset

```python id="1"
X =
[
 [0,0],
 [0,1],
 [1,0],
 [1,1]
]
```

Expected outputs:

```python id="2"
y =
[
 [0],
 [1],
 [1],
 [0]
]
```

---

# Step 2: Initialize Weights

Assume random weights are:

```python id="3"
wh =
[
 [0.2,0.4],
 [0.3,0.1]
]
```

```python id="4"
bh = [[0.1,0.2]]
```

```python id="5"
wo =
[
 [0.5],
 [0.7]
]
```

```python id="6"
bo = [[0.3]]
```

---

# Step 3: Take First Input

```python id="7"
X = [0,0]
```

Target output:

```python id="8"
y = [0]
```

---

# Step 4: Forward Propagation

## Hidden Layer Input

```python id="9"
hidden_input = np.dot(X,wh) + bh
```

Formula:

HiddenInput = XW_h + b_h

Calculation:

[
(0×0.2)+(0×0.3)+0.1 = 0.1
]

[
(0×0.4)+(0×0.1)+0.2 = 0.2
]

So:

```python id="10"
hidden_input = [0.1,0.2]
```

---

# Step 5: Hidden Layer Output

```python id="11"
hidden_output = sigmoid(hidden_input)
```

Formula:

\sigma(x)=\frac{1}{1+e^{-x}}

[
\sigma(0.1)=0.524
]

[
\sigma(0.2)=0.549
]

So:

```python id="12"
hidden_output = [0.524,0.549]
```

---

# Step 6: Output Layer Input

```python id="13"
final_input = np.dot(hidden_output,wo) + bo
```

Calculation:

[
(0.524×0.5)+(0.549×0.7)+0.3
]

[
=0.262+0.384+0.3
]

[
=0.946
]

So:

```python id="14"
final_input = 0.946
```

---

# Step 7: Predicted Output

```python id="15"
predicted_output = sigmoid(final_input)
```

[
\sigma(0.946)=0.720
]

So:

```python id="16"
predicted_output = 0.720
```

---

# Step 8: Error Calculation

```python id="17"
error = y - predicted_output
```

[
0 - 0.720
]

[
=-0.720
]

So:

```python id="18"
error = -0.720
```

---

# Step 9: Output Delta

```python id="19"
d_output = error * sigmoid_derivative(predicted_output)
```

Derivative:

\sigma'(x)=x(1-x)

[
0.720(1-0.720)=0.2016
]

[
-0.720×0.2016=-0.145
]

So:

```python id="20"
d_output = -0.145
```

---

# Step 10: Hidden Layer Error

```python id="21"
error_hidden = d_output.dot(wo.T)
```

[
-0.145 × [0.5,0.7]
]

[
=[-0.072,-0.101]
]

So:

```python id="22"
error_hidden =
[-0.072,-0.101]
```

---

# Step 11: Hidden Delta

```python id="23"
d_hidden = error_hidden * sigmoid_derivative(hidden_output)
```

Suppose:

```python id="24"
d_hidden =
[-0.018,-0.025]
```

---

# Step 12: Update Output Weights

```python id="25"
wo += hidden_output.T.dot(d_output) * learning_rate
```

Learning rate:

```python id="26"
0.5
```

Old weight:

```python id="27"
0.5
```

Updated weight:

```python id="28"
0.462
```

Similarly all weights update.

---

# Step 13: Update Biases

```python id="29"
bo += np.sum(d_output) * learning_rate
```

New bias:

[
0.3 + (-0.145×0.5)
]

[
=0.2275
]

---

# Step 14: Repeat for 10000 Epochs

```python id="30"
for i in range(epochs)
```

Network repeatedly:

* predicts
* calculates error
* updates weights

until it learns XOR.

---

# Step 15: Final Output After Training

Suppose final output becomes:

```python id="31"
[
 [0.02],
 [0.97],
 [0.96],
 [0.03]
]
```

---

# Final Predictions

| Input | Output |
| ----- | ------ |
| [0,0] | 0      |
| [0,1] | 1      |
| [1,0] | 1      |
| [1,1] | 0      |

Network successfully learns XOR using backpropagation.


# Algorithm: Back Propagation Feed-Forward Neural Network for XOR

1. Start

2. Import NumPy library.

3. Define sigmoid activation function:
   \sigma(x)=\frac{1}{1+e^{-x}}

4. Define derivative of sigmoid function:
   \sigma'(x)=x(1-x)

5. Define XOR input dataset:

   ```python id="1"
   X = [[0,0],
        [0,1],
        [1,0],
        [1,1]]
   ```

6. Define expected output:

   ```python id="2"
   y = [[0],
        [1],
        [1],
        [0]]
   ```

7. Initialize random weights and biases:

   * Input layer → Hidden layer weights (`wh`)
   * Hidden layer bias (`bh`)
   * Hidden layer → Output layer weights (`wo`)
   * Output layer bias (`bo`)

8. Set:

   * Learning rate
   * Number of epochs

9. Start training loop.

10. Perform forward propagation:

* Calculate hidden layer input:
  HiddenInput = XW_h + b_h
* Apply sigmoid activation to get hidden layer output.
* Calculate final layer input:
  FinalInput = HiddenOutputW_o + b_o
* Apply sigmoid activation to get predicted output.

11. Calculate error:

```python id="3"
error = y - predicted_output
```

12. Perform backpropagation:

* Calculate output layer delta.
* Calculate hidden layer error.
* Calculate hidden layer delta.

13. Update weights and biases:

* Update output weights (`wo`)
* Update output bias (`bo`)
* Update hidden weights (`wh`)
* Update hidden bias (`bh`)

14. Repeat steps 10–13 until all epochs are completed.

15. Display predicted output after training.

16. Stop.
"""