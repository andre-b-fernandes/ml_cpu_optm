import numpy as np

def gen_random_matrix(n: int, m: int) -> np.ndarray:
    """
    Generate a random matrix of shape (n, m) with float32 elements.
    """
    return np.random.rand(n, m).astype(np.float32)

def generate_random_input_matrices(n: int, m: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two random matrices of shape (n, k) and (m, k).
    """
    return gen_random_matrix(n, k), gen_random_matrix(m, k)

