import numpy as np
from ..utils import sigmoid, cross_entropy_loss
from ..utils import ModelNotTrainedException

class LogisticRegression:
    """Logistic Regression Algorithm
    """

    def __init__(self, alpha = 0.1, max_iter = 100):
        self.coef_ = []
        self.bias_ = 0
        self.alpha = alpha
        self.max_iter = max_iter
        self.costs_ = []
        self.__trained = False

    def __initialize_weights(self, n_features):
        self.coef_ = np.zeros((1, n_features))
        self.bias_ = 0

    def fit(self, X, y):
        """Fit simple logistic regression classifier model & calculate the coefficients.

        Args:
            X (array-like): Training data of shape (k,n) -- matrix.
            y (array-like): Training output of shape (k,) -- vector.

        Returns:
            self : an instance of the class LogisticRegression
        """
        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))

        y = np.array(y, 'float64')

        self.__initialize_weights(X.shape[1])

        for epoch in range(1, self.max_iter+1):
            yhat = sigmoid(np.dot(self.coef_, X.T) + self.bias_)
            cost = cross_entropy_loss(y, yhat)

            #gradients
            dW = (1 / X.shape[0]) * (np.dot(X.T, (yhat - y.T).T))
            db = (1 / X.shape[0]) * (np.sum(yhat - y.T))

            self.coef_ -= self.alpha * (dW.T)
            self.bias_ -= self.alpha * db

            if epoch % 50 == 0: 
                self.costs_.append(cost)

        self.__trained = True
        return self

    def predict(self, X):
        """Predict output using the logistic regression model.

        Args:
            X (array-like): Input data of shape (k, n)

        Returns:
            yhat (array): Predicted output values of shape (k,)
        """
        if(self.__trained == False):
            raise ModelNotTrainedException(self.predict.__name__)
        
        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))

        yhat = sigmoid(np.dot(self.coef_, X.T) + self.bias_)
        yhat[yhat >= 0.5] = 1.0
        yhat[yhat < 0.5] = 0.0

        return yhat

    def evaluate(self, X, y):
        """Evaluate mean accuracy score of the model.

        Args:
            X (array-like): Evaluation data of shape (k, n)
            Y (array-like): Evaluation output of shape (k,)

        Returns:
            Accuracy (float): The mean accuracy score of the model
        """
        if(self.__trained == False):
            raise ModelNotTrainedException(self.predict.__name__)
        
        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))
        y = np.array(y, 'float64')

        yhat = self.predict(X)

        return np.sum(y == yhat) / y.shape[0]

