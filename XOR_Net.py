from SigmoidLayer import SigmoidLayer
from ReluLayer import ReluLayer
from LinearLayer import LinearLayer
import ComputeCost
from LeakyReluLayer import LeakyReluLayer
from TanhLayer import TanhLayer
import numpy as np
import matplotlib.pyplot as plt

"""
Network structure is:
    input -> linear -> tanh -> linear -> sigmoid  = output 
                            OR
    input -> linear -> sigmoid -> linear -> sigmoid = output
"""

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])
X_train = X.T
Y_train = Y.T


Z1 = LinearLayer(input_shape=X_train.shape, n_out=3, init_style="He")
A1 = TanhLayer()
Z2 = LinearLayer(input_shape=(Z1.n_out,Z1.n_in), n_out=1, init_style="He")
A2 = SigmoidLayer()

learning_rate = 0.1
epochs = 5000
costs = []
np.random.seed(0)

# Training loop
for epoch in range(1, epochs+1):
    # Forward prop
    Z1.forward(X_train)
    A1.forward(Z1.Z)
    Z2.forward(A1.A)
    A2.forward(Z2.Z)

    cost, dA2 = ComputeCost.binary_squared_cost(Y=Y_train, Y_hat=A2.A)

    print("Epoch {} Cost: {}".format(epoch, cost))
    costs.append(cost)

    #Backprop
    A2.backward(dA2)
    Z2.backward(A2.dZ)
    A1.backward(Z2.dA_prev)
    Z1.backward(A1.dZ)

    #Update weights and biases
    Z2.Adam_step(learning_rate=learning_rate)
    Z1.Adam_step(learning_rate=learning_rate)

print(A2.A)
plt.plot(range(0, len(costs)), costs)
plt.show()
