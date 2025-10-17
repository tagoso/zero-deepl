# 3.6 Handwriting recognition (forward propagation)

import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# flatten=True => 1-dimensional vector with 784 (= 28×28) elements
# load_mnist(normalize=True) => devide 784 elements by 255 and scale them to 0 to 1

print(type(x_train))  # <class 'numpy.ndarray'>
# x_train is numpy.ndarray not Python built-in library.

print(x_train.shape)  # (60000, 784) Training images
print(t_train.shape)  # (60000,) Training labels
print(x_test.shape)  # (10000, 784) Test images
print(t_test.shape)  # (10000,) Test labels


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img = x_train[0]
img = img.reshape(28, 28)
print(img.shape)  # (28, 28)
img_show(img)  # Popup: 5

label = t_train[0]
print(label)  # 5


# 3.6.2 Neural Network Inference Processing 🐍
#


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


# ⚠️ In this chapter, we skip training part and use the pre-trained sample_weight.pkl file.


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()  # get_data() returns x_test, t_test
print(x)
print(t)
# This is tuple unpacking
# data = get_data()
# x = data[0]
# t = data[1]

network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # get index of the highest propability
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 3.6.3 Batch Processing

x, _ = get_data()
network = init_network()
W1, W2, W3 = network["W1"], network["W2"], network["W3"]
b1, b2, b3 = network["b1"], network["b2"], network["b3"]

# print(x.shape) # (10000, 784)
# print(x[0].shape) # (784,)
# print(W1.shape) # (784, 50)
# print(W2.shape) # (50, 100)
# print(W3.shape) # (100, 10)
print("b1 shape:" + str(b1.shape))  # b1 shape:(50,)
print("b2 shape:" + str(b2.shape))  # b2 shape:(100,)
print("b3 shape:" + str(b3.shape))  # b3 shape:(10,)

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):  # make an integer list range(start, end, step)
    x_batch = x[i : i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i : i + batch_size])

print("Accuracy(range):" + str(float(accuracy_cnt) / len(x)))

# structure to compare the results with answers
y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y == t)

print(np.sum(y == t))
