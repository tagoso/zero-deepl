# 6.2 Initial Value of Weight
#
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


input_data = np.random.randn(1000, 100)  # just generating 1000 dummy data
node_num = 100  # num of nodes (neurons) in each hidden layer
hidden_layer_size = 5
activations = {}  # store the result of activation

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # here randomizing the weight
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # Below is Xavier
    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # Below is He
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
    # w = np.random.randn(node_num, node_num) * 0.0

    a = np.dot(x, w)

    z = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# Plot Histogram
for i, z in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0:
        plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(z.flatten(), 30, range=(0, 1))
plt.show()
