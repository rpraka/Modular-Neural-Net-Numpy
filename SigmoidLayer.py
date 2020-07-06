import numpy as np


class SigmoidLayer:
    """
        Class containing methods executed by sigmoid activation layer in computational graph
        Arguments:
            none    

        Methods:
            forward(Z): Z = raw output of previous layer
            backward(upstream_grad): upstream grad = gradient chained from suceeding layers
    """

    def __init__(self):
        pass

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-1 * Z))

    def backward(self, upstream_grad):
        """
        compute derivative of A = sigmoid(Z)
        dA/dZ = A * (1 - A), using hadamard product
        """
        # use np.multiply to use hadamard product rather than matrix multiplication (*)
        self.dZ = np.multiply(upstream_grad, np.multiply(self.A, (1 - self.A)))
