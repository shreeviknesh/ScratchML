import numpy as np

def euclidian_distance(x1, x2):
    """A function that calculates the euclidian distance between two vectors.
    """
    return np.linalg.norm(x1 - x2)