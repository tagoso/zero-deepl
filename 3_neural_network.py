import numpy as np
import matplotlib.pylab as plt

# there are many kinds of active functions!
#
# 3.2.1 Sigmoid Function
#
# h(x)=1/1+e^(-x)
#

# 3.2.2 first, redefine 0/1 logic gate as step function
#


def step_function_1(x):
    if x > 0:
        return 1
    else:
        return 0


# above, x cannot be NumPy arrays such as step_function(np.array([1.0, 2.0]))
# so...


def step_function_2(x):
    y = x > 0
    return y.astype(np.int)


# x = np.array([-1.0, 1.0, 2.0, 0])
# print(x)

# y as boolean
# y = x > 0
# print(y)

# y_type
# y_t = y.astype(np.int64)
# print(y_t)


# 3.2.3 Graph of Step Function
#
# def step_function3(x):
#     return np.array(x > 0, dtype=np.int64)


# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function3(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)  # specify y axis range
# plt.show()


# 3.2.4 Here is finally Sigmoid Function!
#


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x_sigmoid = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x_sigmoid))

t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)
print(1.0 / t)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # specify y axis range
# plt.show()


# 3.2.7 ReLU Function
#
def relu(x):
    return np.maximum(0, x)


# 3.3 Multi-Dimensional Arrays 🔥
#

A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))  # 1: since one-dimensional array
print(A.shape)  # (4, ): consisting of four elements
print(A.shape[0])  # 4

# test
# one-dimention
a_1 = np.array([1, 2, 3, 4])
print(a_1.shape, a_1.ndim)  # (4,) 1

# two-dimention
b_1 = np.array([[1, 2, 3], [4, 5, 6]])
print(b_1.shape, b_1.ndim)  # (2, 3) 2

# three-dimention
c_1 = np.zeros((2, 3, 4))  # two blocks, three rows, four columns
print(c_1.shape, c_1.ndim)  # (2, 3, 4) 3
print(c_1)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)  # (3, 2) 3x2 array

# 3.3.2 Product of matrix
#
#
A = np.array([[1, 2], [3, 4]])
print(A.shape)  # (2, 2) 2x2 array
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B))
print(np.dot(B, A))  # ≠ np.dot(A, B)

# Different shapes

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)  # (2, 3)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)  # (3, 2)
print(np.dot(A, B))

C = np.array([[1, 2], [3, 4]])
print(C.shape)
print(A.shape)
# print(
#     np.dot(A, C)
# )  # ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)

A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)
B = np.array([7, 8])
print(B.shape)
print(np.dot(A, B))

X = np.array([1, 2])
print(X.shape)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)
Y = np.dot(X, W)
print(Y)

# Above, X is features of one sample, W is weight, Y is hidden layer output


# 3.4.2 Implementation of Signal Transmission in Each Layer 🔥

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(np.dot(X, W1))
print(B1)
print(A1)
print(A1.shape)

# 1️⃣ Signal transmission from the input layer to the first layer

Z1 = sigmoid(A1)
# sigmoid() is defined already above
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

print(A1)  # [0.3, 0.7, 1.1]
print(Z1)  # [0.57444252, 0.66818777, 0.75026011]

# 2️⃣ Signal transmission from Layer 1 to Layer 2

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(W2.shape)  # (3, 2)
print(B2.shape)  # (2,)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 3️⃣ Signal transmission from the second layer to the output layer


def identity_function(x):
    return x


# Just to replace with sigmoid()

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)  # or Y = A3

print(W3.shape)
print(A3)
print(Y)

# 3.4.3 Summary
# Note: here we add weight and bias manually, but in the actual development,
# it will be initialized by libraries such as PyTorch.


def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network


print(init_network())


def forward(network, x):  # look here we use 'forward'... 'backward' will appear later
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

# NN can be used both for classification and regression.
# In general, Classification uses SoftMax, Regression uses Identity Function.


# 3.5
#
