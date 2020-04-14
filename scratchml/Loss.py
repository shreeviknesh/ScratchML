import numpy as np
def mean_squared_error(Y, Yhat):
    """Calculate mean squared error.

    Args:
        Y (array-like): Actual output of the model.
        Yhat (array-like): Predicted output of the model.

    Returns:
        L (float64): The mse between Y and Yhat.
    """
    Y = np.array(Y, 'float64')
    Yhat = np.array(Yhat, 'float64')
    return np.sum((Y - Yhat) ** 2) / Y.shape[0]

def root_mean_squared_error(Y, Yhat):
    """Calculate root mean squared error.

    Args:
        Y (array-like): Actual output of the model.
        Yhat (array-like): Predicted output of the model.

    Returns:
        L (float64): The rmse between Y and Yhat.
    """
    Y = np.array(Y, 'float64')
    Yhat = np.array(Yhat, 'float64')

    return (np.sum((Y - Yhat) ** 2) ** 0.5) / Y.shape[0]

def mean_absolute_error(Y, Yhat):
    """Calculate mean absolute error.

    Args:
        Y (array-like): Actual output of the model.
        Yhat (array-like): Predicted output of the model.

    Returns:
        L (float64): The mae between Y and Yhat.
    """
    Y = np.array(Y, 'float64')
    Yhat = np.array(Yhat, 'float64')

    return np.sum(np.abs(Y - Yhat)) / Y.shape[0]