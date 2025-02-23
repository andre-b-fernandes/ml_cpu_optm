// SSE intrinsics
// 4 bits for SSE, 8 bits for AVX
#ifdef __x86_64__
   #include <immintrin.h>
#else  	
  #include "sse2neon.h"
#endif
// SIMD-optimized Euclidean distance function
void euclidean_distance(float* Q, float* X, float* distances, int M, int N, int D) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m128 sum_vec = _mm_setzero_ps();  // Initialize SIMD sum register to 0

            int k;
	    int block_size = 4;
            for (k = 0; k <= D - block_size; k += block_size) {  // Process 4 elements at a time
                __m128 q_vec = _mm_loadu_ps(&Q[i * D + k]);
                __m128 x_vec = _mm_loadu_ps(&X[j * D + k]);
                __m128 diff = _mm_sub_ps(q_vec, x_vec);
                __m128 squared = _mm_mul_ps(diff, diff);
                sum_vec = _mm_add_ps(sum_vec, squared);
            }

            // Horizontal sum of sum_vec (reduces 4 floats to 1)
            __m128 temp = _mm_add_ps(sum_vec, _mm_movehl_ps(sum_vec, sum_vec));
            temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            float sum = _mm_cvtss_f32(temp);  // Extract final sum

            // Handle remaining elements (if D is not a multiple of 4)
            for (; k < D; k++) {
                float diff = Q[i * D + k] - X[j * D + k];
                sum += diff * diff;
            }


	    // Store result in distance matrix
	    _mm_store_ss(&distances[i * N + j], _mm_sqrt_ps(temp));
        }
    }
}
