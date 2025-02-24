#include <math.h>

// Function to compute pairwise Euclidean distance
void euclidean_distance(float* Q, float* X, float* distances, int M, int N, int D) {
    // Iterate over all pairs of points
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum_sq = 0.0;
            for (int k = 0; k < D; k++) {
	 	// We access Q and X in a row-major order by computing the index
		// as i * D + k and j * D + k respectively.
                float diff = Q[i * D + k] - X[j * D + k];
                sum_sq += diff * diff;
            }
            distances[i * N + j] = sqrt(sum_sq);
        }
    }
}

