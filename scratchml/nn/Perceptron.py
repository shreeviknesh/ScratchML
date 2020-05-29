import numpy as np
from ..utils import ModelNotTrainedException, threshold

class Perceptron:
    """Perceptron implementation.
    """

    def __init__(self, alpha = 0.01, max_iter = 500):
        """Initialize the Perceptron object.

        Args:
            alpha (float): Learning rate of the classifier.
            max_iter (int): Maximum number of iterations.

        Returns:
            self: an instance of class Perceptron.
        """
        self.coef_ = []
        self.bias_ = []
        self.alpha = alpha
        self.max_iter = max_iter
        self.__trained = False

    def train(self, X, y):
        """Train the Perceptron and calculate coefficients.

        Args:
            X (array-like): Training data of shape (k,n) -- matrix.
            y (array-like): Trianing output of shake (k,) -- vector.

        Returns:
            self: an instance of class Perceptron.
        """
        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))
        y = np.array(y, 'float64')

        self.coef_ = np.random.rand(X.shape[1])
        self.bias_ = np.random.random()

        for _ in range(self.max_iter):
            flag = True
            for i in range(X.shape[0]):
                net = np.dot(self.coef_, X[i]) + self.bias_
                yhat = int(threshold(net, 0.5))

                if yhat > y[i]:
                    self.coef_ = self.coef_ - self.alpha * X[i]
                    self.bias_ = self.bias_ - self.alpha
                    flag = False
                    break
                elif yhat < y[i]:
                    self.coef_ = self.coef_ + self.alpha * X[i]
                    self.bias_ = self.bias_ + self.alpha
                    flag = False
                    break

            if flag == True:
                break

        self.__trained = True
        return self

    def predict(self, X):
        """Predict the output using the Perceptron classifier.

        Args:
            X (array-like): Input data of shake (k,n).

        Returns:
            y (array): Predicted output of shape (k,).
        """
        if self.__trained == False:
            raise ModelNotTrainedException(self.predict.__name__)

        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))

        y = self.coef_ * X + self.bias_
        return threshold(y, 0.5)

    def evaluate(self, X, y):
        """Evaluate mean accuracy score of the model.

        Args:
            X (array-like): Evaluation data of shape (k, n).
            Y (array-like): Evaluation output of shape (k,).

        Returns:
            Accuracy (float): The mean accuracy score of the model.
        """
        if(self.__trained == False):
            raise ModelNotTrainedException(self.predict.__name__)

        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))
        y = np.array(y, 'float64')

        yhat = self.predict(X).reshape(X.shape[0],)
        return np.sum(y == yhat) / X.shape[0]
