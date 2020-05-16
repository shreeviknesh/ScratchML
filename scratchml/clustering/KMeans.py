import numpy as np
from ..utils import ModelNotTrainedException, InvalidValueException
from ..utils.distance import euclidean_distance
from copy import deepcopy

class KMeans:
    """KMeans Algorithm for Clustering
    """

    def __init__(self, n_clusters, max_iter = 100):
        """Initialize the KMeans Object.

        Args:
            n_clusters (int): The value of K.

        Returns:
            self: an instance of the class KMeans.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = []
        self.labels_ = []
        self.input_shape = None
        self.__trained = False

    def fit(self, X):
        """Fit the KMeans model.

        Args:
            X (array-like): Training data of shape (k,n) -- matrix.

        Returns:
            self: an instance of the class KMeans.
        """
        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))

        if X.shape[0] < self.n_clusters:
            raise InvalidValueException("The number of clusters cannot exceed the number of samples")
        
        # Choosing k random points as the initial cluster centroids
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        # Iterating for max_iter
        for _ in range(self.max_iter):
            # Calculating the distance of every point to every center
            centroids = np.array([np.argmin([euclidean_distance(x_i, y_k) for y_k in self.centers]) for x_i in X])

            # IF there are not enough clusters generated
            if (len(np.unique(centroids)) < self.n_clusters):
                self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
            else:
                # No centroid has been re-assigned
                if (np.array_equal(self.centers, centroids)):
                    break

                # Assigning the centers to the calculated centroids
                self.centers = [X[centroids == k].mean(axis = 0) for k in range(self.n_clusters)]        

        self.centers = np.array(self.centers)
        self.__trained = True
        self.input_shape = X.shape[1:]
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        """Cluster the given inputs and give the cluster center as output.

        Args:
            X (array-like): Testing data of shape (k,n) -- matrix.

        Returns:
            y (array): Predicted cluster label of shape (k,) -- vector.
        """
        if self.__trained == False:
            raise ModelNotTrainedException(self.predict.__name__)

        X = np.array(X, 'float64')
        if len(X.shape) == 1 or X.shape[1] == 1:
            X = X.reshape((X.shape[0], 1))
        
        if X.shape[1:] != self.input_shape:
            raise InvalidValueException("Input shape does not match trained input shape, retrain the model or use inputs with trained shape")

        y = [np.argmin([euclidean_distance(x, center) for center in self.centers]) for x in X]
        return np.array(y)