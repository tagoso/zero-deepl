import numpy as np
from dataset.mnist import load_mnist

# Fix random number
np.random.seed(0)


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


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dx = dout.copy()
        dout[self.mask] = 0
        dx = dout

        return dx


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


# def numerical_gradient(f, x):
#     h = 1e-4  # 0.0001
#     grad = np.zeros_like(x)

#     it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
#     while not it.finished:
#         idx = it.multi_index
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h
#         fxh1 = f(x)  # f(x+h) f returns scalar

#         x[idx] = tmp_val - h
#         fxh2 = f(x)  # f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2 * h)

#         x[idx] = tmp_val  # put the value back
#         it.iternext()

#     return grad


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # initialize weight
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # generate layers
        self.layers = dict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["ReLU1"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x = input data, t = training data
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # settings
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads

    # x = input data, t = training data
    # Numerical gradient is O(P) (P=Parameter) and heavy: can be used for gradient check

    # def numerical_gradient(self, x, t):
    #     def loss_W(W):
    #         return self.loss(x, t)

    #     grads = {}
    #     grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
    #     grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
    #     grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
    #     grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

    #     return grads


# ——— Load data ———

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size // batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Forward & Loss & Backprop
    grad = network.gradient(x_batch, t_batch)

    # Parameter update
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
