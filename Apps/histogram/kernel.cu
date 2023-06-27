
__global__ 
void histo_simple(  unsigned char *buffer, 
                    long size, 
                    unsigned *histo)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while(i < size){
        // Add atomic (data-race among all the threads in the grid):
        atomicAdd( &(histo[buffer[i]]), 1);
        i += stride;
    }
}

__global__
void histo_privatization(   unsigned char *buffer, 
                            long size, 
                            unsigned *histo)
{

}
