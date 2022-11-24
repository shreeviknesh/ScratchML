import numpy as np

from ..utils import mean_squared_error, mean_absolute_error, root_mean_squared_error
from ..utils import InvalidValueException, ModelNotTrainedException

class SimpleLinearRegression:
    """Simple Linear Regression algorithm.
    """

    def __init__(self):
        self.coef_ = None
        self.__trained = False

    def fit(self, x, y):
        """Fit simple linear regression model & calculate the coefficients.

        Args:
            x (array-like): Training data of shape (k,) 
            y (array-like): Training output of shape (k,) 

        Returns:
            self : an instance of the class SimpleLinearRegression
        """
        x = np.array(x, 'float64')
        y = np.array(y, 'float64')

        xy = x * y
        xx = x * x
        yy = y * y

        a = np.dot(np.sum(y), np.sum(xx)) - np.dot(np.sum(x), np.sum(xy))
        b = x.shape[0] * np.sum(xy) - np.dot(np.sum(x), np.sum(y))
        c = x.shape[0] * np.sum(xx) - np.dot(np.sum(x), np.sum(x))

        self.coef_ = np.array([a/c, b/c])
        self.__trained = True

        return self

    def predict(self, x):
        """Predict output using the linear model.

        Args:
            x (array-like): Input data of shape (k,)

        Returns:
            yhat (array): Predicted output values of shape (k,)
        """
        if(self.__trained == False):
            raise ModelNotTrainedException(self.predict.__name__)
        
        x = np.array(x, 'float64')
        return (self.coef_[1] * x) + self.coef_[0]

    def evaluate(self, x, y, loss='mse'):
        """Evaluate accuracy of the linear model.

        Args:
            x (array-like): Evaluation data of shape (k,)
            y (array-like): Evaluation output of shape (k,)
            loss (string, optional): The loss function to be used -- mse/rmse/mae

        Returns:
            Error (float): The total sum-of-squared-error of the predictions
        """
        if(self.__trained == False):
            raise ModelNotTrainedException(self.evaluate.__name__)

        x = np.array(x, 'float64')
        y = np.array(y, 'float64')
        yhat = self.predict(x)

        available_losses = {
            'mse': mean_squared_error,
            'rmse': root_mean_squared_error,
            'mae': mean_absolute_error
        }
        
        error = 0
        if loss in available_losses.keys():
            error = available_losses[loss](y, yhat)
            deviation_from_mean = available_losses[loss](y, [np.mean(y)] * y.shape[0])

            r2 = (deviation_from_mean - error) / deviation_from_mean

        else:
            message = {
                'expected': available_losses.keys(),
                'recieved': loss
            }
            raise InvalidValueException(message)
        
        return {
            'loss': error,
            'score': r2 
        }

