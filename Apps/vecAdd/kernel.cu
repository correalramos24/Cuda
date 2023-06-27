__global__ void vecAdd( float *C, 
                        const float * __restrict__ A, 
                        const float * __restrict__ B,
                        const unsigned size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i <= size) C[i] = A[i] + B[i];
}