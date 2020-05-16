import numpy as np
import operator
from ..utils import euclidean_distance, ModelNotTrainedException, InvalidValueException

class KNN:
    """K-Nearest Neighbors Algorithm for Classification.
    """

    def __init__(self, num_neighbors):
        """Initialize the KNN Object.

        Args:
            num_neighbors (int): The value of K.

        Returns:
            self : an instance of the class KNN.
        """
        self.num_neighbors = num_neighbors
        self.X = []
        self.y = []
        self.labels = []
        self.__trained = False

    def fit(self, X, y):
        """Fit KNN model.

        Args:
            X (array-like): Training data of shape (k,n) -- matrix.
            y (array-like): Target values of shape (k,) -- vector.

        Returns:
            self : an instance of the class KNN.
        """
        self.X = np.array(X, 'float64')
        self.y = np.array(y)
        
        self.labels = np.unique(self.y)

        num_labels = self.labels.shape[0]
        if num_labels > self.X.shape[0]:
            raise InvalidValueException("The number of classes cannot exceed the number of samples")
        
        # if self.num_neighbors % num_labels == 0:
        #     raise InvalidValueException("K must not be a multiple of the number of classes")

        self.__trained = True
        return self

    def get_class_labels(self, x):
        distances = []        
        for i in range(self.X.shape[0]):
            distances.append((self.y[i], euclidean_distance(x, self.X[i])))
        distances.sort(key=operator.itemgetter(1))
        
        votes = dict()
        for i in range(self.num_neighbors):
            if distances[0] in votes.keys():
                votes[distances[0]] += 1
            else:
                votes[distances[0]] = 1

        maxVotes = 0
        label = None

        for key, val in votes.items():
            if val > maxVotes:
                maxVotes = val
                label = key
        
        return label[0]


    def predict(self, X):
        """Classify the given values into classes.

        Args:
            X (array-like): Testing data of shape (k,n) -- matrix.

        Returns:
            y (array): Predicted class labels of shape (k,) -- vector.
        """
        if self.__trained == False:
            raise ModelNotTrainedException(self.predict.__name__)
        
        X = np.array(X, 'float64')
        if X.shape == ():
            return self.get_class_labels(X)

        y = []
        for x in X:
            y.append(self.get_class_labels(x))
        
        return np.array(y)

    def evaluate(self, X, y):
        """Evaluate the KNN model.

        Args:
            X (array-like): Testing data of shape (k,n) -- matrix.
            y (array-like): Target values of shape (k,) -- vector.

        Returns:
            Accuracy (float): The mean accuracy score of the model
        """
        if(self.__trained == False):
            raise ModelNotTrainedException(self.predict.__name__)
        
        X = np.array(X, 'float64')
        yhat = self.predict(X)
        return np.sum(y == yhat) / len(y)