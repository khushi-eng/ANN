import numpy as np


# ---------------- DIGIT PATTERNS ----------------

# Each number is represented using a 5 × 3 matrix

# 1 = valid point
# 0 = invalid point

digits = [

    # ---------------- DIGIT 0 ----------------
    [
        [1,1,1],
        [1,0,1],
        [1,0,1],
        [1,0,1],
        [1,1,1]
    ],

    
    # ---------------- DIGIT 1 ----------------
    [
        [0,1,0],
        [1,1,0],
        [0,1,0],
        [0,1,0],
        [1,1,1]
    ],

    
    # ---------------- DIGIT 2 ----------------
    [
        [1,1,1],
        [0,0,1],
        [1,1,1],
        [1,0,0],
        [1,1,1]
    ],

    
    # ---------------- DIGIT 3 ----------------
    [
        [1,1,1],
        [0,0,1],
        [1,1,1],
        [0,0,1],
        [1,1,1]
    ],

    
    # ---------------- DIGIT 4 ----------------
    [
        [1,0,1],
        [1,0,1],
        [1,1,1],
        [0,0,1],
        [0,0,1]
    ],

    
    # ---------------- DIGIT 5 ----------------
    [
        [1,1,1],
        [1,0,0],
        [1,1,1],
        [0,0,1],
        [1,1,1]
    ],

    
    # ---------------- DIGIT 6 ----------------
    [
        [1,1,1],
        [1,0,0],
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ],

    
    # ---------------- DIGIT 7 ----------------
    [
        [1,1,1],
        [0,0,1],
        [0,1,0],
        [1,0,0],
        [1,0,0]
    ],

    
    # ---------------- DIGIT 8 ----------------
    [
        [1,1,1],
        [1,0,1],
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ],

    
    # ---------------- DIGIT 9 ----------------
    [
        [1,1,1],
        [1,0,1],
        [1,1,1],
        [0,0,1],
        [1,1,1]
    ]

]



# ---------------- PREPARE INPUT DATA ----------------

# np.array() converts list into numpy array

# flatten() converts 5×3 matrix into single 15-element vector

X = np.array([np.array(d).flatten() for d in digits])



# ---------------- TARGET OUTPUTS ----------------

# np.eye(10) creates 10×10 identity matrix

# Used for one-hot encoding

# Example:
# Digit 0 -> [1 0 0 0 0 0 0 0 0 0]
# Digit 1 -> [0 1 0 0 0 0 0 0 0 0]

y = np.eye(10)



# ---------------- SIGMOID ACTIVATION FUNCTION ----------------

# Sigmoid converts values between 0 and 1

def sigmoid(x):

    # np.exp() calculates exponential value

    return 1 / (1 + np.exp(-x))



# ---------------- SIGMOID DERIVATIVE ----------------

# Used during backpropagation

def sigmoid_derivative(x):

    return x * (1 - x)



# ---------------- INITIALIZE WEIGHTS ----------------

# np.random.rand(rows, columns)

# Creates random values between 0 and 1


# Weight matrix between input layer and hidden layer

# 15 input neurons → 16 hidden neurons

W1 = np.random.rand(15, 16)



# Weight matrix between hidden layer and output layer

# 16 hidden neurons → 10 output neurons

W2 = np.random.rand(16, 10)



# ---------------- LEARNING RATE ----------------

# Controls speed of learning

learning_rate = 0.1



# ---------------- TRAIN NETWORK ----------------

# Number of iterations for training

epochs = 10000



# Training loop

for i in range(epochs):


    # ---------------- FORWARD PROPAGATION ----------------

    # np.dot() performs matrix multiplication

    # Input layer → Hidden layer

    Z1 = np.dot(X, W1)


    # Apply sigmoid activation

    A1 = sigmoid(Z1)



    # Hidden layer → Output layer

    Z2 = np.dot(A1, W2)


    # Final output after activation

    output = sigmoid(Z2)



    # ---------------- ERROR CALCULATION ----------------

    # Difference between actual and predicted output

    error = y - output



    # ---------------- BACKPROPAGATION ----------------

    # Calculate output layer delta

    d_output = error * sigmoid_derivative(output)



    # dot() used for matrix multiplication

    # Calculate hidden layer error

    error_hidden = d_output.dot(W2.T)


    # W2.T means transpose of matrix

    # Rows become columns and columns become rows


    # Calculate hidden layer delta

    d_hidden = error_hidden * sigmoid_derivative(A1)



    # ---------------- UPDATE WEIGHTS ----------------

    # A1.T = transpose of hidden layer output

    W2 += A1.T.dot(d_output) * learning_rate


    # X.T = transpose of input matrix

    W1 += X.T.dot(d_hidden) * learning_rate

# ---------------- TEST INPUT ----------------

# Test digit = 7

test = np.array([

    [1,1,1],
    [0,0,1],
    [0,1,0],
    [1,0,0],
    [1,0,0]

])



# flatten() converts matrix into vector

test_input = test.flatten()



# ---------------- TESTING ----------------

# Forward propagation for test input


# Hidden layer output

hidden = sigmoid(np.dot(test_input, W1))


# Final network output

final_output = sigmoid(np.dot(hidden, W2))



# ---------------- PREDICT DIGIT ----------------

# np.argmax() returns index of largest value

# Largest output neuron represents predicted digit

predicted_digit = np.argmax(final_output)



# ---------------- DISPLAY RESULTS ----------------

print("Test Pattern:\n")

print(test)


print("\nNetwork Output:\n")

print(final_output)


print("\nRecognized Digit:")

print(predicted_digit)

# # Dry Run of Neural Network Digit Recognition Program

# We will do dry run for **one training iteration** and one test example. 

# ---

# # STEP 1: Input Digit Patterns

# Example digit 0:

# ```python id="1"
# [
#  [1,1,1],
#  [1,0,1],
#  [1,0,1],
#  [1,0,1],
#  [1,1,1]
# ]
# ```

# ---

# # STEP 2: Flatten Input

# ```python id="2"
# flatten()
# ```

# Converts 5×3 matrix into 15-element vector.

# Digit 0 becomes:

# ```python id="3"
# [1,1,1,1,0,1,1,0,1,1,0,1,1,1,1]
# ```

# Similarly all digits become vectors.

# So:

# ```python id="4"
# X.shape
# ```

# becomes:

# ```python id="5"
# (10,15)
# ```

# Meaning:

# * 10 digits
# * each digit has 15 inputs

# ---

# # STEP 3: Create Target Output

# ```python id="6"
# y = np.eye(10)
# ```

# Creates identity matrix.

# Example:

# Digit 0 target:

# ```python id="7"
# [1 0 0 0 0 0 0 0 0 0]
# ```

# Digit 1 target:

# ```python id="8"
# [0 1 0 0 0 0 0 0 0 0]
# ```

# ---

# # STEP 4: Initialize Weights

# ```python id="9"
# W1 = np.random.rand(15,16)
# ```

# Suppose first few values are:

# ```python id="10"
# W1 =
# [
#  [0.2,0.5,...],
#  [0.1,0.7,...],
#  ...
# ]
# ```

# Shape:

# ```python id="11"
# (15,16)
# ```

# ---

# ```python id="12"
# W2 = np.random.rand(16,10)
# ```

# Shape:

# ```python id="13"
# (16,10)
# ```

# ---

# # STEP 5: Forward Propagation

# ---

# # Input → Hidden Layer

# ```python id="14"
# Z1 = np.dot(X, W1)
# ```

# Formula:

# Z_1 = XW_1

# Take first digit input:

# ```python id="15"
# [1,1,1,1,0,1,1,0,1,1,0,1,1,1,1]
# ```

# Multiply with weight matrix.

# Suppose hidden neuron 1 calculation:

# [
# (1\times0.2)+(1\times0.1)+(1\times0.4)+...
# ]

# Suppose result:

# ```python id="16"
# 2.3
# ```

# Similarly all 16 hidden neurons calculate outputs.

# Suppose:

# ```python id="17"
# Z1 =
# [2.3,1.8,0.5,...]
# ```

# ---

# # Apply Sigmoid

# ```python id="18"
# A1 = sigmoid(Z1)
# ```

# Formula:

# \sigma(x)=\frac{1}{1+e^{-x}}

# For 2.3:

# [
# \sigma(2.3)=0.91
# ]

# For 1.8:

# [
# \sigma(1.8)=0.85
# ]

# Suppose:

# ```python id="19"
# A1 =
# [0.91,0.85,0.62,...]
# ```

# ---

# # Hidden → Output Layer

# ```python id="20"
# Z2 = np.dot(A1, W2)
# ```

# Again matrix multiplication.

# Suppose:

# ```python id="21"
# Z2 =
# [1.2,0.5,2.1,...]
# ```

# ---

# # Final Output

# ```python id="22"
# output = sigmoid(Z2)
# ```

# Suppose:

# ```python id="23"
# output =
# [0.76,0.62,0.89,0.11,...]
# ```

# These represent prediction strengths.

# ---

# # STEP 6: Error Calculation

# Suppose actual digit = 0

# Correct output:

# ```python id="24"
# [1,0,0,0,0,0,0,0,0,0]
# ```

# Predicted:

# ```python id="25"
# [0.76,0.62,0.89,0.11,...]
# ```

# ---

# ```python id="26"
# error = y - output
# ```

# Result:

# ```python id="27"
# [
# 1-0.76,
# 0-0.62,
# 0-0.89,
# ...
# ]
# ```

# Suppose:

# ```python id="28"
# error =
# [0.24,-0.62,-0.89,...]
# ```

# ---

# # STEP 7: Output Delta

# ```python id="29"
# d_output = error * sigmoid_derivative(output)
# ```

# Sigmoid derivative:

# \sigma'(x)=x(1-x)

# For output 0.76:

# [
# 0.76(1-0.76)
# ]

# [
# =0.1824
# ]

# Multiply with error:

# [
# 0.24 \times 0.1824
# ]

# [
# =0.0437
# ]

# Similarly for all outputs.

# ---

# # STEP 8: Hidden Layer Error

# ```python id="30"
# error_hidden = d_output.dot(W2.T)
# ```

# Error moves backward from output layer to hidden layer.

# ---

# # STEP 9: Hidden Delta

# ```python id="31"
# d_hidden = error_hidden * sigmoid_derivative(A1)
# ```

# Calculates correction for hidden neurons.

# ---

# # STEP 10: Update Weights

# ---

# ## Update W2

# ```python id="32"
# W2 += A1.T.dot(d_output) * learning_rate
# ```

# Suppose:

# ```python id="33"
# learning_rate = 0.1
# ```

# Weights slightly change.

# Example:

# Old weight:

# ```python id="34"
# 0.5
# ```

# New weight:

# ```python id="35"
# 0.504
# ```

# ---

# ## Update W1

# ```python id="36"
# W1 += X.T.dot(d_hidden) * learning_rate
# ```

# Input-hidden weights also improve.

# ---

# # STEP 11: Repeat 10000 Times

# ```python id="37"
# for i in range(epochs):
# ```

# Network repeatedly:

# * predicts
# * calculates error
# * adjusts weights

# Eventually learns patterns.

# ---

# # STEP 12: Testing

# Test digit:

# ```python id="38"
# [
#  [1,1,1],
#  [0,0,1],
#  [0,1,0],
#  [1,0,0],
#  [1,0,0]
# ]
# ```

# This is digit 7.

# Flattened:

# ```python id="39"
# [1,1,1,0,0,1,0,1,0,1,0,0,1,0,0]
# ```

# ---

# # Hidden Layer Output

# ```python id="40"
# hidden = sigmoid(np.dot(test_input,W1))
# ```

# Suppose:

# ```python id="41"
# [0.92,0.11,0.84,...]
# ```

# ---

# # Final Output

# ```python id="42"
# final_output = sigmoid(np.dot(hidden,W2))
# ```

# Suppose:

# ```python id="43"
# [0.01,0.03,0.05,0.02,0.01,0.04,0.08,0.97,0.02,0.01]
# ```

# Largest value:

# ```python id="44"
# 0.97
# ```

# at index:

# ```python id="45"
# 7
# ```

# ---

# # STEP 13: Prediction

# ```python id="46"
# predicted_digit = np.argmax(final_output)
# ```

# Result:

# ```python id="47"
# 7
# ```

# ---

# # Final Output

# ```python id="48"
# Recognized Digit:
# 7

