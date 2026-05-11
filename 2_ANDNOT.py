#PS:Aim: Generate ANDNOT function using McCulloch-Pitts neural net.
# McCulloch-Pitts Neural Network for ANDNOT Function

# ANDNOT means:
# Output is 1 only when:
# x1 = 1 AND x2 = 0

# Truth Table:
# x1  x2   Output
# 0   0      0
# 0   1      0
# 1   0      1
# 1   1      0


# ---------------- INPUT COMBINATIONS ----------------

# Creating all possible binary inputs
# Each tuple represents (x1, x2)

inputs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]


# ---------------- WEIGHTS ----------------

# Weight for x1
# Positive because x1 should support output = 1
w1 = 1

# Weight for x2
# Negative because x2 should prevent output = 1
w2 = -1


# ---------------- THRESHOLD VALUE ----------------

# Threshold decides whether neuron fires or not

# If weighted sum >= threshold
# output = 1

# Else
# output = 0
threshold = 1


# ---------------- MCCULLOCH-PITTS NEURON ----------------

# Loop runs for every input pair
for x1, x2 in inputs:

    # Weighted Sum Formula:
    # net = (input1 × weight1) + (input2 × weight2)

    net = (x1 * w1) + (x2 * w2)


    # Activation Function
    # If net value reaches threshold
    # neuron activates

    if net >= threshold:
        output = 1
    else:
        output = 0


    # Printing result
    print("Input:", x1, x2, "Output:", output)