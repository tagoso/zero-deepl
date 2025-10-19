import numpy as np
import matplotlib.pyplot as plt

# --- Graph ---
x0 = np.linspace(-2, 2, 50)
x1 = np.linspace(-2, 2, 50)
X0, X1 = np.meshgrid(x0, x1)

Z = X0**2 + X1**2

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X0, X1, Z, cmap="viridis", edgecolor="none")

ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_zlabel("f(x0, x1)")
ax.set_title("f(x) = x0^2 + x1^2")

plt.show()


# f(x, y) = x^2 + y^2
def f(x):
    return x[0] ** 2 + x[1] ** 2


def grad_f(x):
    return np.array([2 * x[0], 2 * x[1]])


x = np.array([3.0, 2.0])  # defalut position
eta = 0.1

for i in range(10):
    x = x - eta * grad_f(x)
    print(f"step {i}: x={x}, f={f(x):.4f}")
