import ctypes
import numpy as np

# Load the compiled C shared library
lib = ctypes.CDLL("./naive.so")  # Use ".dll" on Windows

# Define function prototype: void euclidean_distance(float*, float*, float*, int, int, int)
lib.euclidean_distance.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # Q
    ctypes.POINTER(ctypes.c_float),  # X
    ctypes.POINTER(ctypes.c_float),  # D_matrix
    ctypes.c_int,  # M
    ctypes.c_int,  # N
    ctypes.c_int   # D
]

def euclidean_distance(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Call the C function for Euclidean distance computation"""
    M, D = len(Q), len(Q[0])
    N = len(X)

    # Flatten NumPy arrays to 1D C-compatible arrays
    q_flat = Q.flatten()
    x_flat = X.flatten()
    distances = np.zeros((M, N), dtype=np.float32)

    # Convert to C pointers
    q_ptr = q_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    x_ptr = x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    d_ptr = distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the C function
    lib.euclidean_distance(q_ptr, x_ptr, d_ptr, M, N, D)
    return distances
