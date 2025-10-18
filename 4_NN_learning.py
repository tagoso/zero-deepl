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
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)
print(t_train.shape[0])  # 60000


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# Probability of "2" is "0.6"
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_2 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print("SSE_2:", sum_squared_error(np.array(y_2), np.array(t)))
# 0.09750000000000003 Small error

# Probability of "7" is "0.6"
y_7 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print("SSE_7:", sum_squared_error(np.array(y_7), np.array(t)))
# 0.5975 Big error


# 4.2.2 Cross Entropy Error


def cross_entropy_error(y, t):
    delta = 1e-7  # add this to prevent "-inf"
    return -np.sum(t * np.log(y + delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_2 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print("CEE_2", cross_entropy_error(np.array(y_2), np.array(t)))
# Smaller error 0.510825457099338
y_7 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print("CEE_7", cross_entropy_error(np.array(y_7), np.array(t)))
# Bigger error 2.302584092994546


# 4.2.3　Mini-batch learning (sampling in statistics)

# Pick up random 10 samples from x_train dataset

train_size = x_train.shape[0]  # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
# This is Fancy Indexing feature by NumPy.
# batch_mask is something like [11897  1555 40806 46742  1419 31482 21500 33403 14440 58675]
# Then get the actual data by t_train[batch_mask]

print(np.random.choice(60000, 10))


# One-hot
def cross_entropy_error_batch(y, t):  # y is output, t is traing data
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# Not One-hot
def cross_entropy_error_batch2(y, t):
    if (
        y.ndim == 1
    ):  # .ndim is NumPy object ("Number of array dimensions"), so y has to be NumPy array.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


y_nparray = np.array(
    [
        [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
        [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0],
    ]
)
t_nparray = np.array([2, 7])

print(cross_entropy_error_batch2(y_nparray, t_nparray))


# 4.3 Numerical Differentiation
