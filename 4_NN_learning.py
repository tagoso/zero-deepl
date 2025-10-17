# 4.1
# Perceptron Convergence Theorem
# In order to solve XOR problem, one line is not enough.
# So we need MLP or Multi-Layer Perceptron
#
#
# 4.1.1
# Feature and Feature Value
#
# Feature Engineering
#   1. Feature Selection
#   2. Feature Creation / Extraction
#   3. Feature Transformation
#   4. Dimensionality Reduction
#
# 4.2 Loss Function


# 4.2.1 Sum of Squared Error

import numpy as np


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# Probability of "2" is "0.6"
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_2 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(sum_squared_error(np.array(y_2), np.array(t)))

# Probability of "7" is "0.6"
y_7 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(sum_squared_error(np.array(y_7), np.array(t)))

# 4.2.2 Cross Entropy Error
