#. Write a python program to illustrate ART neural network.
# numpy library is used for mathematical operations and arrays

import numpy as np



# ---------------- INITIALIZE WEIGHTS FUNCTION ----------------

# This function creates random weights for ART network

def initialize_weights(input_dim):


    # np.random.uniform(size=(input_dim,))
    # creates random decimal numbers between 0 and 1

    # input_dim tells how many weights are needed

    weights = np.random.uniform(size=(input_dim,))



    # np.sum(weights)
    # adds all weight values together

    # weights /= value
    # divides every weight by total sum

    # this is called normalization

    # normalization keeps values balanced

    weights /= np.sum(weights)



    # return sends weights back to function caller

    return weights




# ---------------- SIMILARITY FUNCTION ----------------

# This function checks how similar input pattern is to weights

def calculate_similarity(input_pattern, weights):


    # np.minimum(a,b)
    # compares both arrays element by element
    # and takes smaller value from each position

    # .sum()
    # adds all those minimum values

    # higher result = more similarity

    return np.minimum(input_pattern, weights).sum()




# ---------------- UPDATE WEIGHTS FUNCTION ----------------

# This function updates weights during learning

def update_weights(input_pattern, weights, vigilance):


    # while True creates infinite loop

    # loop continues until return statement occurs

    while True:



        # calculate similarity between input and weights

        activation = calculate_similarity(

            input_pattern,
            weights

        )



        # ---------------- VIGILANCE TEST ----------------

        # vigilance is minimum similarity needed

        # if similarity is greater than vigilance
        # pattern belongs to same category

        if activation >= vigilance:


            # return updated weights

            return weights



        else:


            # np.argmax(input_pattern)
            # finds index of largest value

            # weight at that position is increased

            weights[np.argmax(input_pattern)] += 1



            # normalize weights again

            weights /= np.sum(weights)




# ---------------- ART NETWORK FUNCTION ----------------

# Main ART neural network function

def ART_neural_network(input_patterns, vigilance):


    # input_patterns.shape gives rows and columns

    # num_patterns = number of patterns

    # input_dim = number of elements in each pattern

    num_patterns, input_dim = input_patterns.shape



    # empty list to store categories/clusters

    categories = []



    # loop through every input pattern

    for pattern in input_patterns:



        # initially no category is matched

        matched_category = None



        # compare pattern with existing categories

        for category in categories:



            # calculate similarity

            similarity = calculate_similarity(

                pattern,
                category["weights"]

            )



            # ---------------- VIGILANCE CHECK ----------------

            # if similarity is enough

            if similarity >= vigilance:


                # pattern belongs to this category

                matched_category = category

                break




        # ---------------- CREATE NEW CATEGORY ----------------

        # if no category matched

        if matched_category is None:



            # create new random weights

            weights = initialize_weights(input_dim)



            # dictionary used to store
            # weights and patterns together

            matched_category = {

                "weights": weights,
                "patterns": []

            }



            # add category into category list

            categories.append(matched_category)




        # store current pattern inside category

        matched_category["patterns"].append(pattern)



        # update category weights

        matched_category["weights"] = update_weights(

            pattern,
            matched_category["weights"],
            vigilance

        )



    # return all learned categories

    return categories




# ---------------- INPUT PATTERNS ----------------

# Binary input patterns

# 1 means active value
# 0 means inactive value

input_patterns = np.array([

    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 1, 0]

])




# ---------------- VIGILANCE PARAMETER ----------------

# controls strictness of matching

# higher vigilance → more categories
# lower vigilance → fewer categories

vigilance = 0.5




# ---------------- TRAIN ART NETWORK ----------------

# function call for ART learning

categories = ART_neural_network(

    input_patterns,
    vigilance

)




# ---------------- DISPLAY RESULTS ----------------

# enumerate() gives index and value together

for i, category in enumerate(categories):



    # print category number

    print("Category", i + 1)



    print("Patterns:")



    # print every pattern stored in category

    for pattern in category["patterns"]:

        print(pattern)




    print("Weights:")



    # print learned weights

    print(category["weights"])



    print()

"""# Dry Run of XOR Backpropagation Program

Take first input:

```python id="1"
X = [0,0]
y = [0]
```

Assume initial weights:

```python id="2"
W1 =
[[0.2,0.4],
 [0.3,0.1]]

W2 =
[[0.5],
 [0.7]]
```

Biases:

```python id="3"
b1 = [[0.1,0.2]]
b2 = [[0.3]]
```

---

# Forward Propagation

```python id="4"
Z1 = np.dot(X,W1) + b1
```

Calculation:

[
(0×0.2)+(0×0.3)+0.1 = 0.1
]

[
(0×0.4)+(0×0.1)+0.2 = 0.2
]

So:

```python id="5"
Z1 = [0.1,0.2]
```

Apply sigmoid:

\sigma(x)=\frac{1}{1+e^{-x}}

```python id="6"
A1 = [0.524,0.549]
```

---

```python id="7"
Z2 = np.dot(A1,W2) + b2
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

```python id="8"
Z2 = 0.946
```

Apply sigmoid:

```python id="9"
output = 0.720
```

---

# Error Calculation

```python id="10"
error = y - output
```

[
0 - 0.720 = -0.720
]

---

# Backpropagation

```python id="11"
d_output = error * sigmoid_derivative(output)
```

Derivative:

\sigma'(x)=x(1-x)

[
0.720(1-0.720)=0.2016
]

[
-0.720 × 0.2016 = -0.145
]

So:

```python id="12"
d_output = -0.145
```

---

```python id="13"
error_hidden = d_output.dot(W2.T)
```

[
-0.145 × [0.5,0.7]
]

[
=[-0.072,-0.101]
]

---

```python id="14"
d_hidden = error_hidden * sigmoid_derivative(A1)
```

Suppose:

```python id="15"
d_hidden = [-0.018,-0.025]
```

---

# Weight Update

```python id="16"
W2 += A1.T.dot(d_output) * 0.1
```

Old weight:

```python id="17"
0.5
```

New weight:

```python id="18"
0.492
```

Similarly all weights update.

---

# Repeat Training

Loop runs:

```python id="19"
10000 times
```

Weights gradually learn XOR pattern.

---

# Final Outputs

Suppose after training:

```python id="20"
[
 [0.02],
 [0.97],
 [0.96],
 [0.03]
]
```

Predictions:

```python id="21"
[0 0] -> 0
[0 1] -> 1
[1 0] -> 1
[1 1] -> 0
```

Network correctly learns XOR.
# Algorithm: Back Propagation Network for XOR Function

1. Start

2. Import NumPy library.

3. Define XOR input dataset:

   ```python id="1"
   X = [[0,0],
        [0,1],
        [1,0],
        [1,1]]
   ```

4. Define target outputs:

   ```python id="2"
   y = [[0],
        [1],
        [1],
        [0]]
   ```

5. Define sigmoid activation function:
   \sigma(x)=\frac{1}{1+e^{-x}}

6. Define sigmoid derivative function:
   \sigma'(x)=x(1-x)

7. Initialize random weights:

   * Input layer → Hidden layer (`W1`)
   * Hidden layer → Output layer (`W2`)

8. Initialize biases:

   * Hidden layer bias (`b1`)
   * Output layer bias (`b2`)

9. Set learning rate and number of epochs.

10. Start training loop.

11. Perform forward propagation:

* Calculate hidden layer input:
  Z_1 = XW_1 + b_1
* Apply sigmoid activation:

  ```python id="3"
  A1 = sigmoid(Z1)
  ```
* Calculate output layer input:
  Z_2 = A_1W_2 + b_2
* Apply sigmoid activation to get final output.

12. Calculate error:

    ```python id="4"
    error = y - output
    ```

13. Perform backpropagation:

    * Calculate output delta.
    * Calculate hidden layer error.
    * Calculate hidden layer delta.

14. Update weights:

    * Update `W2`
    * Update `W1`

15. Update biases:

    * Update `b2`
    * Update `b1`

16. Repeat steps 11–15 until all epochs complete.

17. Display final outputs.

18. Convert outputs into binary predictions:

    * If output ≥ 0.5 → 1
    * Else → 0

19. Display predicted XOR outputs.

20. Stop.

"""