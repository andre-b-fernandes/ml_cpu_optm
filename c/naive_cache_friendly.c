#include <math.h>

#define B 32  // Block size
#define UNROLL_FACTOR 4

// void euclidean_distance(float* Q, float* X, float* distances, int M, int N, int D) {
//     for (int j_block = 0; j_block < N; j_block += B) {  // Process X in blocks
//         int j_end = (j_block + B < N) ? (j_block + B) : N;  // Ensure last block fits
// 
//         for (int i = 0; i < M; i++) {  // Iterate over Q
//            for (int j = j_block; j < j_end; j++) {  // Iterate within block
//                 float sum_sq = 0.0f;
//                 for (int k = 0; k < D; k++) {  // Compute squared distance
//                     float diff = Q[i * D + k] - X[j * D + k];
//                     sum_sq += diff * diff;
//                 }
//                 distances[i * N + j] = sqrtf(sum_sq);  // Store final distance
//             }
//         }
//     }
// }

void euclidean_distance(float* Q, float* X, float* distances, int M, int N, int D) {
    for (int j_block = 0; j_block < N; j_block += B) {  // Process X in blocks
        int j_end = (j_block + B < N) ? (j_block + B) : N;  // Ensure last block fits
	    for (int i = 0; i < M; i++) {
		for (int j = j_block; j < j_end; j++) {  // Iterate within block
		    float sum_sq = 0.0;
		    int k = 0;
		    
		    // Unrolled loop
		    for (; k <= D - UNROLL_FACTOR; k += UNROLL_FACTOR) {
			float diff1 = Q[i * D + k] - X[j * D + k];
			float diff2 = Q[i * D + k + 1] - X[j * D + k + 1];
			float diff3 = Q[i * D + k + 2] - X[j * D + k + 2];
			float diff4 = Q[i * D + k + 3] - X[j * D + k + 3];
			sum_sq += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
		    }
		    
		    // Handle remaining elements
		    for (; k < D; k++) {
			float diff = Q[i * D + k] - X[j * D + k];
			sum_sq += diff * diff;
		    }
		    
		    distances[i * N + j] = sqrt(sum_sq);
		}
	    }
	}
}
