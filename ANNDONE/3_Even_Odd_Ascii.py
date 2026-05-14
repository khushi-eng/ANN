"""Aim: Program using Perceptron Neural Network to recognize even and odd numbers. Given
numbers are in ASCII from 0 to 9"""

import numpy as np

# Taking input from user
j = int(input("Enter a Number (0-9): "))

# Step activation function
step_function = lambda x: 1 if x >= 0 else 0

# Training data using ASCII binary values
training_data = [
    {'input': [1, 1, 0, 0, 0, 0], 'label': 1},  # 48 -> even
    {'input': [1, 1, 0, 0, 0, 1], 'label': 0},  # 49 -> odd
    {'input': [1, 1, 0, 0, 1, 0], 'label': 1},  # 50 -> even
    {'input': [1, 1, 0, 0, 1, 1], 'label': 0},  # 51 -> oddc
    {'input': [1, 1, 0, 1, 0, 0], 'label': 1},  # 52 -> even
    {'input': [1, 1, 0, 1, 0, 1], 'label': 0},  # 53 -> odd
    {'input': [1, 1, 0, 1, 1, 0], 'label': 1},  # 54 -> even
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},  # 55 -> odd
    {'input': [1, 1, 1, 0, 0, 0], 'label': 1},  # 56 -> even
    {'input': [1, 1, 1, 0, 0, 1], 'label': 0},  # 57 -> odd
]

# Initial weights
weights = np.array([0, 0, 0, 0, 0, 1])
print("Initial Weight:",weights)
# Training the perceptron
for data in training_data:
    inp = np.array(data['input'])
    label = data['label']

    output = step_function(np.dot(inp, weights))
    error = label - output

    weights += inp * error

# Convert the entered number into its ASCII value
# Example: '5' -> ASCII = 53
ascii_value = ord(str(j))

# Convert ASCII value into 6-bit binary format
# Example: 53 -> 110101
test_input = np.array([int(x) for x in format(ascii_value, '06b')])

# Calculate weighted sum using dot product
# Apply step activation function
# If output is 0 -> odd
# If output is 1 -> even
result = "odd" if step_function(np.dot(test_input, weights)) == 0 else "even"

# Display final trained weights after learning
print("Final Weight:", weights)