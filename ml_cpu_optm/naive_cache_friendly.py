import numpy as np
from math import sqrt

def euclidean_distance(Q: np.ndarray, X: np.ndarray) -> list[list[float]]:
    """
    Compute the Euclidean distance between two matrices in a naive way.
    Pure python implementation

    Arguments:
        Q -- A matrix of shape (M, D)
        X -- A matrix of shape (N, D)
    Returns: A matrix of shape (M, N) where each element (i, j) is the Euclidean distance between Q[i] and X[j]
    """   
    M, D = len(Q), len(Q[0])  # Q has M rows, D features
    N = len(X)  # X has N rows, D features

    # Initialize distance matrix (M x N)
    B = 128
    distances = [[0.0 for _ in range(N)] for _ in range(M)]
    for j_block in range(0, N, B):  # Process X in blocks of size B
        for i in range(M):  # Iterate over each row of Q
                for j in range(j_block, min(j_block + B, N)):  # Iterate within block
                    dist = 0.0
                    for k in range(D):  # Compute squared distance
                        diff = Q[i][k] - X[j][k]
                        dist += diff * diff
                    distances[i][j] = sqrt(dist)  # Compute final Euclidean distance

    return distances
