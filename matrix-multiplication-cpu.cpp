#include <stdio.h>
#include <stdlib.h>
#include <chrono>

int main() {

    int M = 1024, K = 1024, N = 1024;

    int* A = (int*)malloc(M*K*sizeof(int));
    int* B = (int*)malloc(K*N*sizeof(int));
    int* C = (int*)malloc(M*N*sizeof(int));

    for(int i = 0; i <= M-1; i += 1) {
        for(int k = 0; k <= K-1; k += 1) {
            A[i * K + k] = 1; 
        }
    }

    for(int k = 0; k <= K-1; k += 1) {
        for(int j = 0; j <= N-1; j += 1) {
            B[k * N + j] = 1; 
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i <= M-1; i += 1) {
        for(int j = 0; j <= N-1; j += 1) {
            int sum = 0;
            for(int k = 0; k <= K-1; k += 1) {
                sum += A[i*K+k] * B[k*N+j];
            }
            C[i * N + j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("CPU time: %f ms\n", elapsed.count());

    printf("C[0] is: %d (expected %d)\n", C[0], K);
    printf("C[last] is: %d (expected %d)\n", C[M*N-1], K);

    free(A);
    free(B);
    free(C);

    return 0;
}