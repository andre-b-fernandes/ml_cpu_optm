import numpy as np

def euclidean_distance(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance between two matrices using numpy.
    Pure numpy implementation

    Arguments:
        Q -- A matrix of shape (M, D)
        X -- A matrix of shape (N, D)
    Returns: A matrix of shape (M, N) where each element (i, j) is the Euclidean distance between Q[i] and X[j]
    """   
    distances = np.linalg.norm(Q[:, None] - X, axis=2)
    return distances
