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
