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
