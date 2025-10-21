import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


# 4.4.2 Gradient on NN
#


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
# create an instance like below
# net
# ├─ W : [[ 0.13, -0.52,  0.44],
#          [ 0.92, -0.35,  0.77]]
# ├─ predict(x)
# └─ loss(x, t)

print(net.W)  # weight parameter

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))

t = np.array([0, 0, 1])

print(net.loss(x, t))


def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)

# 4.5 Implementation of Learning Algorithms
