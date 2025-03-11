import ctypes
import numpy as np

# Load the compiled C shared library
lib = ctypes.CDLL("./vectorized_openmp.so")  # Use ".dll" on Windows

# Define function prototype: void euclidean_distance(double*, double*, double*, int, int, int)
lib.euclidean_distance.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # Q
    ctypes.POINTER(ctypes.c_float),  # X
    ctypes.POINTER(ctypes.c_float),  # D_matrix
    ctypes.c_int,  # M
    ctypes.c_int,  # N
    ctypes.c_int   # D
]

def euclidean_distance(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Call the C function for Euclidean distance computation vectorized"""
    M, D = len(Q), len(Q[0])
    N = len(X)

    # Flatten NumPy arrays to 1D C-compatible arrays
    Q_flat = Q.flatten()
    X_flat = X.flatten()
    D_matrix = np.zeros((M, N), dtype=np.float32)

    # Convert to C pointers
    Q_ptr = Q_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    X_ptr = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    D_ptr = D_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the C function
    lib.euclidean_distance(Q_ptr, X_ptr, D_ptr, M, N, D)
    return D_matrix
