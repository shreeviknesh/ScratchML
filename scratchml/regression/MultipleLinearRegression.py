import numpy as np

from ..utils import mean_squared_error, mean_absolute_error, root_mean_squared_error
from ..utils import InvalidValueException, ModelNotTrainedException

class MultipleLinearRegression:
    """Multiple Linear Regression algorithm.
    """

    def __init__(self):
        self.coef_ = np.array([])
        self.epsi_ = np.array([])
        self.__trained = False

    def __prepare_x(self, X):
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))
        return np.column_stack([np.ones((X.shape[0],1)), X])


    def fit(self, X, y):
        """Fit multiple linear regression model & calculate the coefficients.

        Args:
            x (array-like): Training data of shape (k,n) -- matrix.
            y (array-like): Training output of shape (k,) -- vector.

        Returns:
            Self.
        """
        X = self.__prepare_x(np.array(X, 'float64'))
        y = np.array(y, 'float64')
        
        try:
            self.coef_ = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        except np.linalg.LinAlgError:
            self.coef_ = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
        
        self.epsi_ = y - np.dot(X, self.coef_) # The residuals
        self.__trained = True

        return self

    def predict(self, X):
        """Predict output using the linear model.

        Args:
            X (array-like): Input data of shape (k, n)

        Returns:
            yhat (array): Predicted output values of shape (k,)
        """
        if(self.__trained == False):
            raise ModelNotTrainedException(self.predict.__name__)

        X = self.__prepare_x(np.array(X, 'float64'))
        yhat = (np.dot(X, self.coef_)) # + self.epsi_

        if len(X.shape) == 1 or X.shape[1] == 1:
            return yhat[0]

        return yhat

    def evaluate(self, X, y, loss='mse'):
        """Evaluate error & r2 score of the linear model.

        Args:
            X (array-like): Evaluation data of shape (k, n)
            Y (array-like): Evaluation output of shape (k,)
            loss (string, optional): The loss function to be used [mse/rmse/mae]

        Returns:
            Error (float): The total sum-of-squared-error of the predictions
            Score (float): The r2 score of the model
        """
        if(self.__trained == False):
            raise ModelNotTrainedException(self.evaluate.__name__)

        X = np.array(X, 'float64')
        y = np.array(y, 'float64')
        yhat = self.predict(X)

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

