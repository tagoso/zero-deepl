import numpy as np


# 5.1
# Forward propagation ➡️➡️➡️
# Backward propagation ⬅️⬅️⬅️
#
# 5.2
# sensitivity to some change, kinda partial derivative or gradient
# this idea is related to propagation or chain rule
#
# chain rule
#
# composition of functions
#
# when
# z = t^2
# t = x + y
#
# ∂z/∂x = ∂z/∂t * ∂t/∂x
#
# 5.3
# This is VJP (Vector-Jacobian Product), and what 'autograd' in PyTorch does.
#
# 5.4.1 Implement multiplication ✖️ layer
#
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):  # dout: derivative of output
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


# example of apples

apple_price = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price_sum = mul_apple_layer.forward(apple_price, apple_num)
price_with_tax = mul_tax_layer.forward(apple_price_sum, tax)

print(price_with_tax)  # 220

# backward

print(mul_apple_layer.x)  # 100
print(mul_apple_layer.y)  # 2
print(mul_tax_layer.x)  # 200
print(mul_tax_layer.y)  # 1.1

# Upstream gradient of the final output (∂L/∂price_with_tax)
dprice = 1

# Pass the upstream gradient to the tax layer
# dout here is dprice = ∂L/∂price_with_tax
dapple_price, dtax = mul_tax_layer.backward(dprice)

# Pass the upstream gradient from the tax layer
# dout here is dapple_price = ∂L/∂apple_price_sum
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)

# 5.4.2 Implement Addition ➕ layer


class AddLayer:
    def __init__(self):
        pass  # do nothing

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)  # 1
orange_price = mul_orange_layer.forward(orange, orange_num)  # 2
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # 3
price = mul_tax_layer.forward(all_price, tax)  # 4

print(price)

# backward
dprice = 1
# Backward functions takes one input and emit two outputs!
dall_price, dtax = mul_tax_layer.backward(dprice)  # 4
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # 3
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # 2
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # 1

print(
    dapple_num,  # 110
    dapple,  # 2.2
    dorange_num,  # 165
    dorange,  # 3.3
    dorange_price,  # 1.1
    dapple_price,  # 1.1
    dtax,  # 650
    dall_price,  # 1.1
)

# 5.5 Implementation of Activation Function Layers
#
# 5.5.1 ReLU Layer
#


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = x <= 0
print(mask)
print(mask.dtype)


# 5.5.2 Sigmoid Layer
#
#
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


# 5.6 Implementation of Affine/Softmax Layers
# 5.6.1 Affine Layer
# In Chapter 3...
X = np.random.rand(2)
W = np.random.rand(2, 3)
B = np.random.rand(3)

print(X.shape)
print(W.shape)
print(B.shape)

print(X)
print(W)
print(B)

Y = np.dot(X, W) + B

# 5.6.2 Batch Affine Layer

X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])

print("X_dot_W: ", X_dot_W)  # X_dot_W:  [[ 0  0  0] \ [10 10 10]]
print("X_dot_W + B: ", X_dot_W + B)  # X_dot_W + B:  [[ 1  2  3] \ [11 12 13]]

# During backpropagation, backpropagated values for each data point must be aggregated into the bias element

dY = np.array([[1, 2, 3], [4, 5, 6]])
print(dY)  # [[1 2 3] \ [4 5 6]]

dB = np.sum(dY, axis=0)
print(dB)  # [5 7 9]


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


# 5.6.3 Softmax-with-Loss Layer


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)  # Prevent overflow
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    # y, t: (N, C) one-hot
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # When the training data is in one-hot vector format, convert it to the index of the correct label.
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # Loss
        self.y = None  # output of softmax
        self.t = None  # One-hot vector training data

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(
        self, dout=1
    ):  # dout is fixed since the output is scalar in Softmax + Loss.
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # This is the key. (yi - ti)/N

        return dx


# 5.7.1
