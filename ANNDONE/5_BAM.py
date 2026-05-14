"""Aim:Write a python Program for Bidirectional Associative Memory with two pairs of vectors."""
import numpy as np

# Define first input-output pattern pair
x1 = np.array([1, 1, 1, -1])
y1 = np.array([1, -1])

# Define second input-output pattern pair
x2 = np.array([-1, -1, 1, 1])
y2 = np.array([-1, 1])

# Compute BAM weight matrix
# Outer product creates associative connections
# Final weight matrix stores both memories
W = np.outer(y1, x1) + np.outer(y2, x2)

# Define BAM recall function
def bam(x):

    # Calculate net output using weight matrix
    # Formula: y = W x
    y = np.dot(W, x)

    # Apply bipolar activation function
    # Positive values become 1
    # Negative values become -1
    y = np.where(y >= 0, 1, -1)

    # Return recalled output pattern
    return y

# Define test input pattern
x_test = np.array([1, -1, -1, -1])

# Recall associated output using BAM
y_test = bam(x_test)

# Display input pattern
print("Input x: ", x_test)

# Display recalled output pattern
print("Output y: ", y_test)



# ## Dry Run of BAM Program

# ### Step 1: Import NumPy

# ```python
# import numpy as np
# ```

# NumPy library is imported for matrix and array operations.

# ---

# # Step 2: Define Pattern Pairs

# ## First Pattern Pair

# ```python
# x1 = np.array([1, 1, 1, -1])
# y1 = np.array([1, -1])
# ```

# Input vector:
# [
# x_1 = [1,\ 1,\ 1,\ -1]
# ]

# Output vector:
# [
# y_1 = [1,\ -1]
# ]

# ---

# ## Second Pattern Pair

# ```python
# x2 = np.array([-1, -1, 1, 1])
# y2 = np.array([-1, 1])
# ```

# Input vector:
# [
# x_2 = [-1,\ -1,\ 1,\ 1]
# ]

# Output vector:
# [
# y_2 = [-1,\ 1]
# ]

# ---

# # Step 3: Compute Weight Matrix

# The BAM weight matrix is calculated using:

# W = y_1x_1^T + y_2x_2^T

# ---

# ## Find Outer Product of First Pair

# ```python
# np.outer(y1, x1)
# ```

# Calculation:

# [
# \begin{bmatrix}
# 1\
# -1
# \end{bmatrix}
# \times
# \begin{bmatrix}
# 1 & 1 & 1 & -1
# \end{bmatrix}
# ]

# Result:

# [
# \begin{bmatrix}
# 1 & 1 & 1 & -1\
# -1 & -1 & -1 & 1
# \end{bmatrix}
# ]

# ---

# ## Find Outer Product of Second Pair

# ```python
# np.outer(y2, x2)
# ```

# Calculation:

# [
# \begin{bmatrix}
# -1\
# 1
# \end{bmatrix}
# \times
# \begin{bmatrix}
# -1 & -1 & 1 & 1
# \end{bmatrix}
# ]

# Result:

# [
# \begin{bmatrix}
# 1 & 1 & -1 & -1\
# -1 & -1 & 1 & 1
# \end{bmatrix}
# ]

# ---

# ## Add Both Outer Products

# ```python
# W = np.outer(y1, x1) + np.outer(y2, x2)
# ```

# [
# W =
# \begin{bmatrix}
# 1 & 1 & 1 & -1\
# -1 & -1 & -1 & 1
# \end{bmatrix}
# +
# \begin{bmatrix}
# 1 & 1 & -1 & -1\
# -1 & -1 & 1 & 1
# \end{bmatrix}
# ]

# Final Weight Matrix:

# [
# W =
# \begin{bmatrix}
# 2 & 2 & 0 & -2\
# -2 & -2 & 0 & 2
# \end{bmatrix}
# ]

# ---

# # Step 4: Define BAM Recall Function

# ```python
# def bam(x):
# ```

# Function accepts input vector `x`.

# ---

# ## Multiply Weight Matrix with Input Vector

# ```python
# y = np.dot(W, x)
# ```

# Formula:

# genui{"math_block_widget_always_prefetch_v2":{"content":"y = Wx"}}

# ---

# # Step 5: Test Input Pattern

# ```python
# x_test = np.array([1, -1, -1, -1])
# ```

# Test Input:

# [
# x = [1,\ -1,\ -1,\ -1]
# ]

# ---

# # Step 6: Calculate Output

# ## Matrix Multiplication

# [
# y =
# \begin{bmatrix}
# 2 & 2 & 0 & -2\
# -2 & -2 & 0 & 2
# \end{bmatrix}
# \times
# \begin{bmatrix}
# 1\
# -1\
# -1\
# -1
# \end{bmatrix}
# ]

# ---

# ## First Row Calculation

# [
# (2\times1) + (2\times-1) + (0\times-1) + (-2\times-1)
# ]

# [
# = 2 -2 +0 +2
# ]

# [
# = 2
# ]

# ---

# ## Second Row Calculation

# [
# (-2\times1) + (-2\times-1) + (0\times-1) + (2\times-1)
# ]

# [
# = -2 +2 +0 -2
# ]

# [
# = -2
# ]

# ---

# ## Net Output Vector

# [
# y = [2,\ -2]
# ]

# ---

# # Step 7: Apply Bipolar Activation Function

# ```python
# y = np.where(y >= 0, 1, -1)
# ```

# Rule:

# * Positive value → 1
# * Negative value → -1

# So,

# [
# [2,\ -2]
# \rightarrow
# [1,\ -1]
# ]

# Final Output:

# [
# y = [1,\ -1]
# ]

# ---

# # Step 8: Display Output

# ```python
# print("Input x: ", x_test)
# print("Output y: ", y_test)
# ```

# Output:

# ```python
# Input x:  [ 1 -1 -1 -1]
# Output y:  [ 1 -1]
# ```

# ---

# # Final Conclusion

# The BAM network successfully recalls the associated output pattern:

# [
# [1,\ -1,\ -1,\ -1]
# \rightarrow
# [1,\ -1]
# ]

# This shows how Bidirectional Associative Memory stores and retrieves associated vector pairs.
