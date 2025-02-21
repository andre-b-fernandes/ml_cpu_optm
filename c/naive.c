#include <math.h>

// Function to compute pairwise Euclidean distance
void euclidean_distance(float* Q, float* X, float* distances, int M, int N, int D) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum_sq = 0.0;
            for (int k = 0; k < D; k++) {
                float diff = Q[i * D + k] - X[j * D + k];
                sum_sq += diff * diff;
            }
            distances[i * N + j] = sqrt(sum_sq);
        }
    }
}
