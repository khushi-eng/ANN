"""PS:With a suitable example demonstrate the perceptron learning law with its decision regions using
python. Give the output in graphical form."""

import numpy as np
import matplotlib.pyplot as plt

# Input data for AND gate
# Possible combinations of two binary inputs
X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])

# Target output values
# -1 represents FALSE
#  1 represents TRUE
Y = np.array([-1, -1, -1, 1])

# Initialize weights with zeros
# Since there are 2 inputs, 2 weights are needed
w = np.zeros(X.shape[1])

# Initialize bias as 0
b = 0

# Train perceptron for 6 iterations (epochs)
for _ in range(6):

    # Traverse each training sample
    for i in range(X.shape[0]):

        # Calculate weighted sum
        # Formula: (x1*w1 + x2*w2 + b)
        net_input = np.dot(X[i], w) + b

        # Apply sign activation function
        # If value >= 0 → output = 1
        # Else → output = -1
        y_pred = np.sign(net_input)

        # Check whether prediction is wrong
        if y_pred != Y[i]:

            # Update weights using perceptron learning rule
            # w(new) = w(old) + learning_rate * target * input
            w += 0.3 * Y[i] * X[i]

            # Update bias
            b += 0.3 * Y[i]

# Define graph boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Create mesh grid points for plotting decision regions
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict output for every point in mesh grid
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)

# Reshape result according to mesh grid shape
Z = Z.reshape(xx.shape)

# Plot decision boundary and regions
plt.contourf(xx, yy, Z, alpha=0.8)

# Plot training data points
plt.scatter(X[:, 0], X[:, 1], c=Y)

# Label x-axis
plt.xlabel('X1')

# Label y-axis
plt.ylabel('X2')

# Graph title
plt.title('Perceptron Decision Regions')

# Display graph
plt.show()