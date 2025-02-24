import numpy as np
import math
from numba import njit

@njit
def euclidean_distance(Q: np.ndarray, X: np.ndarray) -> list[list[float]]:
    """Computes Euclidean distance using a cache-friendly approach in pure Python"""
    M = len(Q)
    N = len(X)
    D = len(Q[0])
    
    Q_sq = [sum([q[d] ** 2 for d in range(D)]) for q in Q]
    X_sq = [sum([x[d] ** 2 for d in range(D)]) for x in X]
    distances = [[0.0] * N for _ in range(M)]
    # Transpose X for better cache locality
    X_T = [[X[n][d] for n in range(N)] for d in range(D)]
        
    for i in range(M):
        for j in range(N):
            dot_product = sum([Q[i][d] * X_T[d][j] for d in range(D)])
            distances[i][j] = math.sqrt(Q_sq[i] - 2 * dot_product + X_sq[j])

    return distances
