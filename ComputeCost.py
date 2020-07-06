import numpy as np


def binary_squared_cost(Y, Y_hat):
    """
    Function to compute cost of single or multiclass classification problem,
     using categorical cross entropy cost
    Cost = 1/(2*n_examples) * sum((Y - Y_hat)**2))
    Arguments:
        Y: integer labels of training data
        Y_hat: predicted 
    Returns:
        cost: categorical cross entropy cost
        dY_hat = dCost/dY_hat, derivative of cost function with respect to Y_hat

        to do: implement regularization and dropout**
    """
    n_examples = Y.shape[1]

    # use np.squeeze to flatten resultant matrix/vector to scalar
    cost = np.squeeze(1 / (2 * n_examples) * np.sum(np.square(Y - Y_hat)))

    # C = 1/(2*n_examples) * sum((Y - Y_hat)**0.5)), dC/dY_hat = -1/m * (Y - Y_hat)
    dY_hat = -1/n_examples * (Y - Y_hat)

    return cost, dY_hat
