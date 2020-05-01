import numpy as np
from ..utils import ModelNotTrainedException

class SVM:
    """Support Vector Machine Algorithm for Classification
    """

    def __init__(self, alpha = 0.01, C = 100):
        """Initialize the SVM object.

        Args:
            alpha (float): The learning rate of the classifier.
            C (float): The inverse of the regularization parameter.

        Returns:
            self : an instance of the class SVM.
        """
        self.coef_ = []
        self.bias_ = 0
        self.alpha = alpha
        self.C = C
        self.costs_ = []
        self.__trained = False

    def __initialize_weights(self, n_features):
        self.coef_ = np.random.uniform(-1, 1, (n_features))
        self.bias_ = np.random.uniform(-1, 1, 1)

    def fit(self, X, y, epochs = 100):
        """Fit Support Vector Classifier model & calculate the coefficients.

        Args:
            X (array-like): Training data of shape (k,n) -- matrix.
            y (array-like): Training output of shape (k,) -- vector.
            epochs (int): The number of epochs to train the model.

        Returns:
            self : an instance of the class SVM.
        """
        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))
        y = np.array(y, 'float64')

        self.__initialize_weights(X.shape[1])

        self.epochs = epochs
        for _ in range(1, self.epochs+1):
            yhat = np.dot(X, self.coef_) + self.bias_

            temp = y * yhat
            flag = y * yhat
            flag[temp < 1] = 1
            flag[temp >= 1] = 0

            # Vectorized try
            # self.coef_ += (-2 * self.alpha * (1/self.C) * self.coef_) + flag * X * y.reshape(y.shape[0], 1)
            # self.bias_ += (-2 * self.alpha * (1/self.C) * self.coef_) + flag * X * y.reshape(y.shape[0], 1)

            for i in range(len(y)):
                if y[i] * yhat[i] < 1:
                    self.coef_ += self.alpha * (np.dot(y[i], X[i]) - 2 * (1 / self.C) * self.coef_)
                    self.bias_ += self.alpha * (np.dot(y[i], 1) - 2 * (1 / self.C) * self.bias_)
                else:
                    self.coef_ *= 1 - (2 * self.alpha * (1 / self.C))
                    self.bias_ *= 1 - (2 * self.alpha * (1 / self.C))

        self.__trained = True
        return self

    def predict(self, X):
        """Predict output using the Support Vector Machine classifier

        Args:
            X (array-like): Input data of shape (k, n)

        Returns:
            yhat (array): Predicted output values of shape (k,)
        """
        if self.__trained == False:
            raise ModelNotTrainedException(self.predict.__name__)
        
        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))

        yhat = np.sign(np.dot(X, self.coef_) + self.bias_)
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