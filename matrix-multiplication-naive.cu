#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrix_multiplication(int *dev_A, int *dev_B, int *dev_C, int M, int K, int N){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    if (row < M) {
        if (col < N) {
            for(int k = 0; k <= K-1; k+=1) {
                sum += dev_A[row*K+k] * dev_B[k*N+col];
            }
            dev_C[row * N + col] = sum;
        }
    }

}

int main(){
    int M = 1024, K = 1024, N = 1024;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

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

    int *dev_A;
    int *dev_B;
    int *dev_C;

    cudaMalloc(&dev_A, M*K *sizeof(int));
    cudaMalloc(&dev_B, K*N *sizeof(int));
    cudaMalloc(&dev_C, M*N *sizeof(int));

    cudaMemcpy(dev_A, A, M*K*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, K*N*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrix_multiplication<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, M, K, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Accelerated Matrix Multiplication: NAIVE VERSION\n");
    printf("GPU kernel time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, dev_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("C[0] = %d (expected %d)\n", C[0], K);
    printf("C[last] = %d (expected %d)\n", C[M*N - 1], K);  

    free(A);
    free(B);
    free(C);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return(0);
}