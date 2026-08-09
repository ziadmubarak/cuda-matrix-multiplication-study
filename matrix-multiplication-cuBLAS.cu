#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(){
    cublasHandle_t handle; cublasCreate(&handle);
    float alpha=1.0; float beta=0.0;

    int M = 1024, K = 1024, N = 1024;

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

    float *dev_A;
    float *dev_B;
    float *dev_C;

    cudaMalloc(&dev_A, M*K *sizeof(float));
    cudaMalloc(&dev_B, K*N *sizeof(float));
    cudaMalloc(&dev_C, M*N *sizeof(float));

    cudaMemcpy(dev_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha, 
        dev_B, N,
        dev_A, K,
        &beta,
        dev_C, N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Accelerated Matrix Multiplication: cuBLAS Version\n");
    printf("GPU kernel time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, dev_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("C[0] = %f (expected %d)\n", C[0], K);
    printf("C[last] = %f (expected %d)\n", C[M*N - 1], K);  

    free(A);
    free(B);
    free(C);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return(0);
}