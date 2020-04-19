import numpy as np
def mean_squared_error(y, yhat):
    """Calculate mean squared error.

    Args:
        y (array-like): Actual output of the model.
        yhat (array-like): Predicted output of the model.

    Returns:
        L (float64): The mse between y and yhat.
    """
    y = np.array(y, 'float64')
    yhat = np.array(yhat, 'float64')
    return np.sum((y - yhat) ** 2) / y.shape[0]

def root_mean_squared_error(y, yhat):
    """Calculate root mean squared error.

    Args:
        y (array-like): Actual output of the model.
        yhat (array-like): Predicted output of the model.

    Returns:
        L (float64): The rmse between y and yhat.
    """
    y = np.array(y, 'float64')
    yhat = np.array(yhat, 'float64')

    return (np.sum((y - yhat) ** 2) ** 0.5) / y.shape[0]

def mean_absolute_error(y, yhat):
    """Calculate mean absolute error.

    Args:
        y (array-like): Actual output of the model.
        yhat (array-like): Predicted output of the model.

    Returns:
        L (float64): The mae between y and yhat.
    """
    y = np.array(y, 'float64')
    yhat = np.array(yhat, 'float64')

    return np.sum(np.abs(y - yhat)) / y.shape[0]

def cross_entropy_loss(y, yhat):
    """Calculate cross entropy loss.

    Args:
        y (array-like): Actual output of the model.
        yhat (array-like): Predicted output of the model.

    Returns:
        L (float64): The cross entropy loss between y and yhat.
    """

    # To avoid divide by zero erros (which occur when val == 0 or 1, we replace it with 0+ & 1-)
    y[y == 1] = 1 - 10 ** -10
    y[y == 0] = 10 ** -10

    yhat[yhat == 1] = 1 - 10 ** -10
    yhat[yhat == 0] = 10 ** -10
    
    return -np.mean((y * np.log(yhat) + (1 - y) * np.log(1 - yhat)))