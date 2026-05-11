"""Write a Python program to plot a few activation functions that are being used in neural
networks."""
# Importing numpy library
# numpy is used for mathematical calculations and handling arrays
import numpy as np

# Importing matplotlib library for drawing graphs
import matplotlib.pyplot as plt


# np.linspace(start, end, number_of_values)
# It creates equally spaced values between -10 and 10
# Here we are creating 100 input values
x = np.linspace(-10, 10, 100)


# ---------------- SIGMOID ACTIVATION FUNCTION ----------------

# Formula of Sigmoid Function:
# 1 / (1 + e^(-x))

# np.exp(-x) calculates e^(-x)
# e is Euler's number

# Sigmoid converts input values between 0 and 1
# Mostly used in binary classification problems
sigmoid = 1 / (1 + np.exp(-x))


# ---------------- TANH ACTIVATION FUNCTION ----------------

# np.tanh(x) calculates tanh values

# Tanh function gives output between -1 and 1
# It is centered around zero
# Better than sigmoid in many neural network cases
tanh = np.tanh(x)


# ---------------- ReLU ACTIVATION FUNCTION ----------------

# ReLU stands for Rectified Linear Unit

# np.maximum(0, x)
# compares 0 and x value

# If x is negative -> output becomes 0
# If x is positive -> output remains x

# Most commonly used activation function in deep learning
relu = np.maximum(0, x)


# ---------------- CREATING GRAPH WINDOW ----------------

# figsize sets size of graph window
# (10,6) means width = 10 and height = 6
plt.figure(figsize=(10, 6))


# ---------------- PLOTTING SIGMOID GRAPH ----------------

# plt.plot(x_axis_values, y_axis_values, label_name)

# x contains input values
# sigmoid contains output values

# label is used to show function name in legend
plt.plot(x, sigmoid, label='Sigmoid')


# ---------------- PLOTTING TANH GRAPH ----------------
plt.plot(x, tanh, label='Tanh')


# ---------------- PLOTTING ReLU GRAPH ----------------
plt.plot(x, relu, label='ReLU')


# ---------------- ADDING TITLE ----------------

# Title shown at top of graph
plt.title("Activation Functions in Neural Networks")


# ---------------- ADDING X-AXIS LABEL ----------------

# Label for horizontal axis
plt.xlabel("Input")


# ---------------- ADDING Y-AXIS LABEL ----------------

# Label for vertical axis
plt.ylabel("Output")


# ---------------- ADDING GRID ----------------

# grid(True) adds background lines
# Makes graph easier to read
plt.grid(True)


# ---------------- SHOWING LEGEND ----------------

# legend() displays labels of all functions
plt.legend()


# ---------------- DISPLAYING GRAPH ----------------

# show() finally displays graph window
plt.show()