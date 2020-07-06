import numpy as np


class LeakyReluLayer:
    """
        Class containing methods executed by Leaky ReLU activation layer in computational graph
        Instead of clamping negative elements to zero, remaps them to a small negative number
        Helps to reduce occurence of dead gradients (0)
        Arguments:
            none
        Methods:
            forward(Z): Z = raw output of previous layer
            backward(upstream_grad): upstream grad = gradient chained from suceeding layers
    """

    def forward(self, Z):
        """
        ReLU(Z) = 0 if Z <= 0, = Z if Z > 0
        """
        self.Z = Z
        self.A = np.maximum(0.1*Z, Z)

    def backward(self, upstream_grad):
        """
        compute derivative of A = ReLU(Z)
        dA/dZ = 0.1 if Z<= 0, = 1 if Z > 0
        """
        self.dZ = np.multiply(upstream_grad, np.where(self.Z <= 0, 0.1, 1))
