import numpy as np

def sigmoid(X):
    """Calculate sigmoid activation function.

    Args:
        X (array-like): Input to the function.

    Returns:
        Y (array): The sigmoid activated values of the inputs.
    """
    return 1 / (1 + np.exp(-X))

def relu(X):
    """Calculate ReLU activation function.

    Args:
        X (array-like): Input to the function.

    Returns:
        Y (array): The ReLU activated values of the inputs.
    """
    return np.maximum(0, X)

def softmax(X):
    """Calculate softmax activation function.

    Args:
        X (array-like): Input to the function.

    Returns:
        Y (array): The softmax activated values of the inputs.
    """
    num = np.exp(X)
    den = np.sum(np.exp(X))
    return num / den

def threshold(X, thresh):
    """Calculates the threshold activation function (either 0 or 1).

    Args:
        X (array-like): Input to the function.
        thresh (float): Value of the threshold.

    Returns:
        Y (array): The threshold activated values of the inputs.
    """
    Y = np.array(X)
    Y[Y >= thresh] = 1
    Y[Y < thresh] = 0
    return Y
