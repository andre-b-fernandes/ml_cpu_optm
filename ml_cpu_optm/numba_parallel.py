import numpy as np
from numba import njit, prange

@njit(parallel=True)
def euclidean_distance_parallel(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    M, D = Q.shape
    N, _ = X.shape
    D_matrix = np.empty((M, N), dtype=np.float32)  # Allocate output matrix

    for i in prange(M):  # Parallel loop
        for j in range(N):
            sum_sq = 0.0
            for k in range(D):  # Compute squared difference
                diff = Q[i, k] - X[j, k]
                sum_sq += diff * diff
            D_matrix[i, j] = np.sqrt(sum_sq)  # Compute final distance

    return D_matrix
