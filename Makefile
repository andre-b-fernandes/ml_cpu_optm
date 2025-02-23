naive_c:
	gcc --shared -fPIC -O0 -o naive.so c/naive.c
naive_loop_unrolling_c:
	gcc --shared -fPIC -O0 -o naive_loop_unrolling.so c/naive_loop_unrolling.c
vectorized_c_macos:
	gcc --shared -fPIC -O2 -march=armv8-a -lm -o vectorized.so c/vectorized.c
vectorized_cache_friendly_loop_unrolling_c_macos:
	gcc --shared -fPIC -O2 -march=armv8-a -lm -o vectorized_cache_friendly_loop_unrolling.so c/vectorized_cache_friendly_loop_unrolling.c
vectorized_c_linux:
	gcc --shared -fPIC -O2 -mavx -lm -o vectorized.so c/vectorized.c
vectorized_cache_friendly_loop_unrolling_c_linux:
	gcc --shared -fPIC -O2 -mavx -lm -o vectorized_cache_friendly_loop_unrolling.so c/vectorized_cache_friendly_loop_unrolling.c
all: naive_c naive_loop_unrolling_c vectorized_c_macos vectorized_cache_friendly_loop_unrolling_c_macos
