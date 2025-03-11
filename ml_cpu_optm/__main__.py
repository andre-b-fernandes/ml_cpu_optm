from pyperf import Runner as PerfRunner

from ml_cpu_optm.data import generate_random_input_matrices
from ml_cpu_optm.naive import euclidean_distance as naive_euclidean_distance
from ml_cpu_optm.naive_cache_friendly import euclidean_distance as naive_cache_friendly
from ml_cpu_optm.naive_without_waste import euclidean_distance as naive_without_waste_euclidean_distance
from ml_cpu_optm.naive_cache_friendly_without_waste import euclidean_distance as naive_cache_friendly_without_waste_euclidean_distance
from ml_cpu_optm.naive_loop_unrolling_without_waste import euclidean_distance as naive_loop_unrolling_without_waste_euclidean_distance
from ml_cpu_optm.numpy_vectorized import euclidean_distance as numpy_vectorized_euclidean_distance
from ml_cpu_optm.vectorized_without_waste import euclidean_distance as vectorized_without_waste_euclidean_distance
from ml_cpu_optm.numba_optimization import euclidean_distance as numba_optimization
from ml_cpu_optm.vectorized_cache_friendly_loop_unrolling_without_waste import euclidean_distance as vectorized_cache_friendly_loop_unrolling_without_waste
from ml_cpu_optm.numba_parallel import euclidean_distance_parallel as numba_parallel
from ml_cpu_optm.vectorized_without_waste_openmp import euclidean_distance as vectorized_without_waste_openmp


if __name__ == "__main__":
    runner = PerfRunner()
    M, N, D = 1024, 512, 256
    Q, X = generate_random_input_matrices(M, N, D)
    # runner.bench_func("naive_euclidean_distance", naive_euclidean_distance, Q, X)
    #runner.bench_func("naive_without_waste_euclidean_distance", naive_without_waste_euclidean_distance, Q, X)
    #numba_optimization(Q, X) # call once to compile then it will be faster since cached
    #runner.bench_func("numba_optimization", numba_optimization, Q, X)
    # runner.bench_func("naive_cache_friendly", naive_cache_friendly, Q, X)
    # runner.bench_func("naive_cache_friendly_without_waste_euclidean_distance", naive_cache_friendly_without_waste_euclidean_distance, Q, X)
    # runner.bench_func("naive_loop_unrolling_without_waste_euclidean_distance", naive_loop_unrolling_without_waste_euclidean_distance, Q, X)
    # runner.bench_func("numpy_raw_euclidean_distance", numpy_vectorized_euclidean_distance, Q, X)
    #runner.bench_func("vectorized_without_waste_euclidean_distance", vectorized_without_waste_euclidean_distance, Q, X)
    # runner.bench_func("vectorized_cache_friendly_loop_unrolling_without_waste", vectorized_cache_friendly_loop_unrolling_without_waste, Q, X)
    result1 = numba_parallel(Q, X) # call once to compile then it will be faster since cached
    result2 = vectorized_without_waste_openmp(Q, X) # call once to compile then it will be faster
    runner.bench_func("numba_parallel", numba_parallel, Q, X)
    runner.bench_func("vectorized_without_waste_openmp", vectorized_without_waste_openmp, Q, X)
