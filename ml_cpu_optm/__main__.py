from pyperf import Runner as PerfRunner

from ml_cpu_optm.data import generate_random_input_matrices
from ml_cpu_optm.naive import euclidean_distance as naive_euclidean_distance


if __name__ == "__main__":
    runner = PerfRunner()
    M, N, D = 100, 100, 100
    Q, X = generate_random_input_matrices(M, N, D)
    runner.bench_func("naive_euclidean_distance", naive_euclidean_distance, Q, X)
