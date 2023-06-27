#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "kernel.h"
#include "../common/file.h"

void checkCudaError(const cudaError_t err);

int main(int argc, char const *argv[]){

    float *A_h, *B_h, *C_h, *A_d, *B_d, *C_d;
    unsigned int vec_size;
    char infileA[] = "inputs/input1.dat";
    char infileB[] = "inputs/input2.dat";
    char outfile[] = "output/out.dat";
    size_t vec_bytes;
    
    dim3 dim_grid, dim_block;
    cudaError_t err;

    //1. Allocate host memory for the input/output vectors
    readVector(infileA, &A_h, &vec_size);
    readVector(infileB, &B_h, &vec_size);
    vec_bytes = sizeof(float) * vec_size;
    C_h = (float*) malloc(vec_bytes);
    printf("Load vectors with size %i (bytes %lu)\n", vec_size, vec_bytes);

    //2. Allocate device memory for the input/output vectors
    err = cudaMalloc(&A_d, vec_bytes);
    checkCudaError(err);
    err = cudaMalloc(&B_d, vec_bytes);
    checkCudaError(err);
    err = cudaMalloc(&C_d, vec_bytes);
    checkCudaError(err);

    //3. Copy the input vectors from the host memory to the device memory
    err = cudaMemcpy(A_d, A_h, vec_bytes, cudaMemcpyHostToDevice);
    checkCudaError(err);
    err = cudaMemcpy(B_d, B_h, vec_bytes, cudaMemcpyHostToDevice);
    checkCudaError(err);
    err = cudaMemset(C_d, 0, vec_bytes);
    checkCudaError(err);

    //4. Initialize thread block and kernel grid dimensions
    dim_grid = dim3(ceil(vec_size/512.0));
    dim_block = dim3(512);

    //Invoke CUDA kernel
    vecAdd<<<dim_grid, dim_block>>>(C_d, A_d, B_d, vec_size);
    checkCudaError(cudaGetLastError());

    //Copy the result back to the host & print result
    err = cudaMemcpy(C_h, C_d, vec_bytes, cudaMemcpyDeviceToHost);
    writeVectorFloat(outfile, C_h, vec_size);

    //Free device memory allocations
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    free(A_h); free(B_h); free(C_h);

    return 0;
}


void checkCudaError(const cudaError_t err){
    if(err == cudaSuccess) return;
    printf("%s : %s", cudaGetErrorName(err), cudaGetErrorString(err));
    exit(4);
}