#include <stdio.h>
#include <stdlib.h>
#include <chrono>

extern "C" {
#include "cblas.h"
}

int main(){
    int M = 1024, K = 1024, N = 1024;
    float alpha=1.0; float beta=0.0;

    float* A = (float*)malloc(M*K*sizeof(float));
    float* B = (float*)malloc(K*N*sizeof(float));
    float* C = (float*)malloc(M*N*sizeof(float));

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

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            1.0f,
            A, K,
            B, N,
            0.0f,
            C, N);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("CPU time: %f ms\n", elapsed.count());

    printf("C[0] is: %f (expected %d)\n", C[0], K);
    printf("C[last] is: %f (expected %d)\n", C[M*N-1], K);

    free(A);
    free(B);
    free(C);

    return 0;
}
