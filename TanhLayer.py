import numpy as np


class TanhLayer:
    """
        Class containing methods executed by tanh activation layer in computational graph
        Arguments:
            none
        Methods:
            forward(Z): Z = raw output of previous layer
            backward(upstream_grad): upstream grad = gradient chained from suceeding layers
    """
    
    def forward(self, Z):
        """
        compute tanh(Z)
        """
        self.Z = Z
        self.A = np.tanh(Z)

    def backward(self, upstream_grad):
        """
        compute derivative of A = tanh(Z)
        dA/dZ = sech**2(Z) = 1/cosh**2(Z)
        """
        self.dZ = np.multiply(upstream_grad, np.cosh(self.Z)**-2)
