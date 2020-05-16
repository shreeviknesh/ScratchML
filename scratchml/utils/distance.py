import numpy as np

def euclidean_distance(x1, x2):
    """A function that calculates the euclidian distance between two vectors.
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.linalg.norm(x1 - x2)