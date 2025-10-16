# NN can be used both for classification and regression.
# In general, Classification uses SoftMax, Regression uses Identity Function.

# 3.5

import numpy as np

a = np.array([0.3, 2.9, 4.0])
print(a)

exp_a = np.exp(a)  # Conver real number to probability & make gradient smooth
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

# Turn above into a function softmax(a)


def softmax_overflaw(a):  # This function does not cover overflaw issue.
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# print(softmax_overflaw(a))

#
# 3.5.2 softmax_overflaw(a) can overflow. Here is a solution...

a = np.array([1010, 1000, 990])
# np.exp(a) / np.sum(
#     np.exp(a)
# )  # Calculation of Softmax, it will return warning and 'nan'

c = np.max(a)  # 1010
print(a - c)
print("raw: ", np.exp(a - c) / np.sum(np.exp(a - c)))

# Turn above into the updated softmax()


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


print("softmax　function: ", softmax(a))


# 3.5.3 Features of SoftMax

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)  # [0.01821127 0.24519181 0.73659691] meaning 1.8%, 24.5%, 73.7%
print(np.sum(y))  # the sum will be 1
