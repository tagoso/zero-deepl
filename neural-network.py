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
