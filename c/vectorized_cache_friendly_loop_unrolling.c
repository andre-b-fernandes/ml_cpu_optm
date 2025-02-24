#ifdef __x86_64__
   #include <immintrin.h>
#else  	
  #include "sse2neon.h"
#endif

#include <math.h>       // For sqrtf

void euclidean_distance(float* Q, float* X, float* distances, int M, int N, int D) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m128 sum_vec = _mm_setzero_ps();  // Initialize SIMD sum register to 0
            int k;

            // Unroll loop: process 16 elements at a time
            for (k = 0; k <= D - 16; k += 16) {
                __m128 q1 = _mm_loadu_ps(&Q[i * D + k]);
                __m128 q2 = _mm_loadu_ps(&Q[i * D + k + 4]);
                __m128 q3 = _mm_loadu_ps(&Q[i * D + k] + 8);
                __m128 q4 = _mm_loadu_ps(&Q[i * D + k + 16]);

                __m128 x1 = _mm_loadu_ps(&X[j * D + k]);
                __m128 x2 = _mm_loadu_ps(&X[j * D + k + 4]);
                __m128 x3 = _mm_loadu_ps(&X[j * D + k + 8]);
                __m128 x4 = _mm_loadu_ps(&X[j * D + k + 16]);

                __m128 d1 = _mm_sub_ps(q1, x1);
                __m128 d2 = _mm_sub_ps(q2, x2);
                __m128 d3 = _mm_sub_ps(q3, x3);
                __m128 d4 = _mm_sub_ps(q4, x4);

                sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(d1, d1));
                sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(d2, d2));
                sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(d3, d3));
                sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(d4, d4));

                // Prefetch next elements of X into cache
                _mm_prefetch((const char*)&X[j * D + k + 32], _MM_HINT_T0);
            }

            // Handle remaining elements
            for (; k < D; k++) {
                float diff = Q[i * D + k] - X[j * D + k];
                sum_vec = _mm_add_ss(sum_vec, _mm_set_ss(diff * diff));
            }

            // Horizontal sum of sum_vec
            __m128 temp = _mm_add_ps(sum_vec, _mm_movehl_ps(sum_vec, sum_vec));
            temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            float sum = _mm_cvtss_f32(temp);

            distances[i * N + j] = sqrtf(sum);  // Compute Euclidean distance
        }
    }
}

