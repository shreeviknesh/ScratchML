import numpy as np
class SimpleLinearRegression:
    """Simple Linear Regression algorithm.
    """

    def __init__(self):
        self.coeffs_ = np.array([])

    def fit(self, X, Y):
        """Fit linear model & calculate the co-efficients.

        Args:
            X (numpy array-like): Training data.
            Y (numpy array-like): Training output.

        Returns:
            None.
        """
        X = np.array(X, 'float64')
        Y = np.array(Y, 'float64')

        XY = X * Y
        XX = X * X
        YY = Y * Y

        a = np.dot(np.sum(Y), np.sum(XX)) - np.dot(np.sum(X), np.sum(XY))
        b = X.shape[0] * np.sum(XY) - np.dot(np.sum(X), np.sum(Y))
        c = X.shape[0] * np.sum(XX) - np.dot(np.sum(X), np.sum(X))

        self.coeffs_ = np.array([a/c, b/c])
    
    def calculate_sse(self, Y, Yhat):
        diff = Y - Yhat
        return np.sum(np.dot(diff, diff))

    def predict(self, X):
        """Predict output using the linear model.

        Args:
            X (array-like): Input data.

        Returns:
            Yhat (array): Predicted output values.
        """
        X = np.array(X, 'float64')
        return (self.coeffs_[1] * X) + self.coeffs_[0]

    def evaluate(self, X, Y):
        """Evaluate accuracy of the linear model.

        Args:
            X (array-like): Evaluation data.
            Y (array-like): Evaluation output.

        Returns:
            sse (float): The total sum-of-squared-error of the predictions.
        """
        X = np.array(X, 'float64')
        Y = np.array(Y, 'float64')

        Yhat = self.predict(X)
        return self.calculate_sse(Y, Yhat)