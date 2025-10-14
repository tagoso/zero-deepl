import numpy as np

# 2.3

# 2.3.1


def AND_test(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# 2.3.2 introducing bias (b)

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
print(w * x)  # array([ 0. , 0.5])
print(np.sum(w * x))  # 0.5
print(np.sum(w * x) + b)  # -0.199999...6

# 2.3.3 using bias


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print(AND(0, 0))  # 0 expected
print(AND(1, 0))  # 0 expected
print(AND(0, 1))  # 0 expected
print(AND(1, 1))  # 1 expected
print(AND(4, -2))  # 1 expected

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))

# note: input is not limited to 0/1, so this is not strictly logic gate.

# step function


def step(a: float) -> int:
    return 1 if a > 0 else 0


def gate(x1, x2, w, b) -> int:
    x = np.array([x1, x2], dtype=float)
    w = np.array(w, dtype=float)
    return step(np.dot(w, x) + b)


def AND_S(x1, x2):
    return gate(x1, x2, [0.5, 0.5], -0.7)


def NAND_S(x1, x2):
    return gate(x1, x2, [-0.5, -0.5], 0.7)


def OR_S(x1, x2):
    return gate(x1, x2, [0.5, 0.5], -0.2)


def XOR_S(x1, x2):
    return AND_S(NAND_S(x1, x2), OR_S(x1, x2))


print(XOR_S(1, 0))
