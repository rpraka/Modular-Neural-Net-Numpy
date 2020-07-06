import numpy as np
from utils import initialize_params


class LinearLayer:
    """
        Class containing methods executed by linear layer in computational graph
        Arguments:
            input_shape: shape of output from previous layer
            n_out: shape of output from this linear layer
            init_style: style to initialize weights and bias, can be "default" or "Xavier"

        Methods:
            forward(A_prev): A_prev = activations of previous layer
            backward(upstream_grad): upstream grad = gradient chained from suceeding layers
            compute_step(learning_rate, descent_style)
    """

    def __init__(self, input_shape, n_out, init_style="default"):
        self.n_in = input_shape[1]
        self.n_out = n_out
        # initialize params and get dictionary containing them
        self.params = initialize_params(
            input_shape[0], n_out, init_style=init_style)

    def forward(self, A_prev):
        self.A_prev = A_prev  # store activations from previous layer as attribute

        self.Z = self.params["W"] @ A_prev + self.params["b"]  # Raw output of this layer

    def backward(self, upstream_grad):
        # dL/dW by chaining local and upstream gradient
        # Z = W @ A_prev + b => dZ/dW = A_prev.T
        self.dW = upstream_grad @ self.A_prev.T

        # dL/dA by chaining local and upstream gradient
        # Z = W @ A_prev + b => dZ/dA = W.T
        self.dA_prev = self.params["W"].T @ upstream_grad

        # dL/db by chaining local and upstream gradient
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)
    def batch_GD_step(self, learning_rate):
        self.params["W"] = self.params["W"] - learning_rate * self.dW
        self.params["b"] = self.params["b"] - learning_rate * self.db

    def momentum_step(self, learning_rate, beta=0.9):
        """
        Compute update using exponentially weighted average of previous dW to dampen oscillations in dW
         Arguments:
            learning_rate

            beta: hyperparamter to control extent to which momentum is conserved, 0.9 is approx. averaging over past 10 steps

            bias_correction: boost velocity during initial updates (to be implemented)**
        """
        V_dW = self.params["V_dW"]
        V_dW = beta * V_dW + (1 - beta) * self.dW
        self.params["V_dW"] = V_dW

        V_db = self.params["V_db"]
        V_db = beta * V_db + (1-beta)* self.db
        self.params["V_db"] = V_db

        self.params["W"] = self.params["W"] -\
            learning_rate * self.params["V_dW"]
        self.params["b"] = self.params['b'] -\
            learning_rate * self.params["V_db"]

    def RMS_prop_step(self, learning_rate, beta=0.9, epsilon=10e-8):
        """
        compute update using exponentially weighted average of previous dW to speed up updates in direction where
        dW was small and slow updates in direction where dW was large
         Arguments:
            learning_rate

            beta: hyperparamter to control extent to which momentum is conserved, 0.9 is
            approx. averaging over past 10 steps

            bias_correction: boost velocity during initial updates (to be implemented)**
            epsilon: small number added in update formula to avoid divison by zero
        """

        S_dW = self.params["S_dW"]
        S_dW = beta * S_dW + (1 - beta) * np.square(self.dW)
        self.params["S_dW"] = S_dW

        S_db = self.params["S_db"]
        S_db = beta * S_db + (1 - beta) * np.square(self.db)
        self.params["S_db"] = S_db

        # small number epsilon is addded in denominator to avoid division by zero
        self.params["W"] = self.params["W"] - learning_rate * \
            self.dW/np.sqrt(self.params["S_dW"]+epsilon)
        self.params["b"] = self.params['b'] - learning_rate * \
            self.params["b"] / np.sqrt(self.params["S_db"] + epsilon)

    def Adam_step(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=10e-8):
        """
        combination of momentum and RMS_prop methods of enhancing gradient descent
        Arguments:
            learning_rate

            beta1: hyperparameter associated with momentum component of Adam

            beta2: hyperparameter associated with RMS_prop component of Adam

            epsilon: small number added in update formula to avoid divison by zero
        """
        V_dW = self.params["V_dW"]
        V_dW = beta1 * V_dW + (1 - beta1) * self.dW
        self.params["V_dW"] = V_dW

        V_db = self.params["V_db"]
        V_db = beta1 * V_db + (1-beta1)*self.db
        self.params["V_db"] = V_db

        S_dW = self.params["S_dW"]
        S_dW = beta2 * S_dW + (1 - beta2) * np.square(self.dW)
        self.params["S_dW"] = S_dW

        S_db = self.params["S_db"]
        S_db = beta2 * S_db + (1 - beta2) * np.square(self.db)
        self.params["S_db"] = S_db

        # small number epsilon is addded in denominator to avoid division by zero
        self.params["W"] = self.params["W"] - learning_rate * \
            self.params["V_dW"]/np.sqrt(self.params["S_dW"]+epsilon)
        self.params["b"] = self.params['b'] - learning_rate * \
            self.params["V_db"] / np.sqrt(self.params["S_db"] + epsilon)
