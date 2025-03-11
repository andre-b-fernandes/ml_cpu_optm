naive_c:
	clang --shared -fPIC -O0 -o naive.so c/naive.c
naive_c_optimized:
	clang --shared -fPIC -O3 -o naive.so c/naive.c
naive_loop_unrolling_c:
	clang --shared -fPIC -O0 -o naive_loop_unrolling.so c/naive_loop_unrolling.c
naive_cache_friendly_c:
	clang --shared -fPIC -O0 -o naive_cache_friendly.so c/naive_cache_friendly.c
vectorized_c_macos:
	clang --shared -fPIC -O3 -march=armv8-a -lm -o vectorized.so c/vectorized.c
vectorized_cache_friendly_loop_unrolling_c_macos:
	clang --shared -fPIC -O3 -march=armv8-a -lm -o vectorized_cache_friendly_loop_unrolling.so c/vectorized_cache_friendly_loop_unrolling.c
vectorized_c_linux:
	clang --shared -fPIC -O3 -mavx -lm -o vectorized.so c/vectorized.c
vectorized_cache_friendly_loop_unrolling_c_linux:
	clang --shared -fPIC -O3 -mavx -lm -o vectorized_cache_friendly_loop_unrolling.so c/vectorized_cache_friendly_loop_unrolling.c
vectorized_openmp_c_macos:
	clang --shared -fPIC -fopenmp -O3 -march=armv8-a -lm -L /opt/homebrew/opt/libomp/lib -I /opt/homebrew/opt/libopm/include -o vectorized_openmp.so c/vectorized_openmp.c

all: naive_c naive_cache_friendly_c naive_loop_unrolling_c vectorized_c_macos vectorized_cache_friendly_loop_unrolling_c_macos vectorized_openmp_c_macos
