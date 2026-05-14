import numpy as np

# Define 4 bipolar vectors
vectors = np.array([
    [1, 1, -1, -1],
    [-1, -1, 1, 1],
    [1, -1, 1, -1],
    [-1, 1, -1, 1]
])

# Number of neurons
n = vectors.shape[1]

# Initialize weight matrix
weights = np.zeros((n, n))

# Training using Hebbian rule
for v in vectors:
    weights += np.outer(v, v)

# Remove self-connections
np.fill_diagonal(weights, 0)

print("Weight Matrix:")
print(weights)

# Activation function
def activation(x):
    return np.where(x >= 0, 1, -1)

# Hopfield recall function
def hopfield(input_vector, weights):
    net_input = np.dot(weights, input_vector)
    output_vector = activation(net_input)
    return output_vector

# Test with stored vector
input_vector = [1,-1,1,-1]

output_vector = hopfield(input_vector, weights)

print("\nInput vector:")
print(input_vector)

print("\nOutput vector:")
print(output_vector)


"""# Dry Run of Hopfield Network Program

---

# Step 1: Define Bipolar Vectors

```python id="1"
vectors =
[
 [1, 1,-1,-1],
 [-1,-1, 1, 1],
 [1,-1, 1,-1],
 [-1,1,-1,1]
]
```

These patterns are stored in Hopfield Network.

---

# Step 2: Number of Neurons

```python id="2"
n = vectors.shape[1]
```

Each vector has 4 elements.

So:

```python id="3"
n = 4
```

Meaning:

* 4 neurons

---

# Step 3: Initialize Weight Matrix

```python id="4"
weights = np.zeros((n,n))
```

Creates:

```python id="5"
weights =
[
 [0,0,0,0],
 [0,0,0,0],
 [0,0,0,0],
 [0,0,0,0]
]
```

---

# Step 4: Training Using Hebbian Rule

Formula:

W = \sum xx^T

---

# First Vector

```python id="6"
v = [1,1,-1,-1]
```

Outer product:

```python id="7"
np.outer(v,v)
```

Calculation:

[
\begin{bmatrix}
1\
1\
-1\
-1
\end{bmatrix}
\times
\begin{bmatrix}
1 & 1 & -1 & -1
\end{bmatrix}
]

Result:

```python id="8"
[
 [ 1, 1,-1,-1],
 [ 1, 1,-1,-1],
 [-1,-1, 1, 1],
 [-1,-1, 1, 1]
]
```

Add to weights:

```python id="9"
weights =
[
 [ 1, 1,-1,-1],
 [ 1, 1,-1,-1],
 [-1,-1, 1, 1],
 [-1,-1, 1, 1]
]
```

---

# Second Vector

```python id="10"
[-1,-1,1,1]
```

Outer product gives same matrix.

Add again:

```python id="11"
weights =
[
 [ 2, 2,-2,-2],
 [ 2, 2,-2,-2],
 [-2,-2, 2, 2],
 [-2,-2, 2, 2]
]
```

---

# Third Vector

```python id="12"
[1,-1,1,-1]
```

Outer product:

```python id="13"
[
 [ 1,-1, 1,-1],
 [-1, 1,-1, 1],
 [ 1,-1, 1,-1],
 [-1, 1,-1, 1]
]
```

Add:

```python id="14"
weights =
[
 [ 3, 1,-1,-3],
 [ 1, 3,-3,-1],
 [-1,-3, 3, 1],
 [-3,-1, 1, 3]
]
```

---

# Fourth Vector

```python id="15"
[-1,1,-1,1]
```

Outer product same as third.

Final weights:

```python id="16"
weights =
[
 [ 4, 0, 0,-4],
 [ 0, 4,-4, 0],
 [ 0,-4, 4, 0],
 [-4, 0, 0, 4]
]
```

---

# Step 5: Remove Self Connections

```python id="17"
np.fill_diagonal(weights,0)
```

Diagonal becomes zero.

Final weight matrix:

```python id="18"
weights =
[
 [ 0, 0, 0,-4],
 [ 0, 0,-4, 0],
 [ 0,-4, 0, 0],
 [-4, 0, 0, 0]
]
```

---

# Step 6: Test Input Vector

```python id="19"
input_vector = [1,-1,1,-1]
```

---

# Step 7: Calculate Net Input

```python id="20"
net_input = np.dot(weights,input_vector)
```

Formula:

Net = WX

---

## First Neuron

[
(0×1)+(0×-1)+(0×1)+(-4×-1)
]

[
=4
]

---

## Second Neuron

[
(0×1)+(0×-1)+(-4×1)+(0×-1)
]

[
=-4
]

---

## Third Neuron

[
(0×1)+(-4×-1)+(0×1)+(0×-1)
]

[
=4
]

---

## Fourth Neuron

[
(-4×1)+(0×-1)+(0×1)+(0×-1)
]

[
=-4
]

So:

```python id="21"
net_input = [4,-4,4,-4]
```

---

# Step 8: Apply Activation Function

```python id="22"
activation(net_input)
```

Rule:

* positive → 1
* negative → -1

Result:

```python id="23"
output_vector = [1,-1,1,-1]
```

---

# Final Output

```python id="24"
Input Vector:
[1,-1,1,-1]

Output Vector:
[1,-1,1,-1]
```

Hopfield network successfully recalls stored pattern.
"""