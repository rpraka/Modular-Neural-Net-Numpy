import numpy as np


def initialize_params(n_in, n_out, init_style="default"):
    """
    Function to initialize weights and bias of linear layer which calls this function

    Arguments:
        n_in: number of incoming inputs
        n_out: number of outputs
        init_style: style of weight initialization

    Returns:
        params: dictionary of randomly initialized weights ("W") and bias("b") matrices

    """

    params = dict()
    params["b"] = np.zeros((n_out, 1))  # bias is initialized to vector of 0s
    params["V_dW"] = 0  # Initial velocity used in momentum update for W
    params["V_db"] = 0  # Initial velocity used in momentum update for b
    params["S_dW"] = 0  # Initial sum used in RMS_prop update for W
    params["S_db"] = 0  # Initial sum used in RMS_prop update for b

    def default_init():
        """
        Init weights from 0-centered normal distribution with small variance (0.01**2)
        """
        params["W"] = np.random.randn(n_out, n_in) * 0.01
        # sample weights from normal distribution, mean: 0, var:0.01**2
        return params

    def He_init():
        """
        Intialization strategy to preserve distribution of weights throughout depth of network for ReLU activation
        """
        # sample weights from normal distribution, mean: 0, var: 2/n_in
        params["W"] = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)

        return params

    def Xavier_init():
        """
        Another intialization strategy to preserve distribution of weights throughout depth of network for ReLU activation
        """
        # sample weights from normal distribution, mean: 0, var: 1/n_in
        params["W"] = np.random.randn(n_out, n_in) * np.sqrt(1 / n_in)

        return params

    return eval(init_style + "_init()")
