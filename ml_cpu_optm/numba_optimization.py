import numpy as np
import math
from numba import njit, prange

@njit
def euclidean_distance(Q: np.ndarray, X: np.ndarray) -> list[list[float]]:
    """Computes Euclidean distance using a cache-friendly approach in pure Python"""
    M = len(Q)
    N = len(X)
    D = len(Q[0])
    
    Q_sq = [sum([q[d] ** 2 for d in prange(D)]) for q in Q]
    X_sq = [sum([x[d] ** 2 for d in prange(D)]) for x in X]
    distances = [[0.0] * N for _ in prange(M)]
    # Transpose X for better cache locality
    X_T = [[X[n][d] for n in prange(N)] for d in prange(D)]
        
    for i in prange(M):
        for j in prange(N):
            dot_product = sum([Q[i][d] * X_T[d][j] for d in prange(D)])
            distances[i][j] = math.sqrt(Q_sq[i] - 2 * dot_product + X_sq[j])

    return distances
