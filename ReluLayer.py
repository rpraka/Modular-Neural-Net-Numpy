import numpy as np


class ReluLayer:
    """
        Class containing methods executed by ReLU activation layer in computational graph
        Arguments: none

        Methods:
            forward(Z): Z = raw output of previous layer
            backward(upstream_grad): upstream grad = gradient chained from suceeding layers
    """

    def forward(self, Z):
        """
        ReLU(Z) = 0 if Z <= 0, = Z if Z > 0
        """
        self.Z = Z
        self.A = np.maximum(0, Z)

    def backward(self, upstream_grad):
        """
        compute derivative of A = ReLU(Z)
        dA/dZ = 0 if Z <= 0, = 1 if Z > 0
        """
        self.dZ = np.multiply(upstream_grad, np.where(self.Z <= 0, 0, 1))

