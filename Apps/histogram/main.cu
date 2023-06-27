#include <stdio.h>

#include "../common/file.h"
#include "kernel.h"

#define MAX_VAL 256
#define BLOCKS 32
#define THREADS 256

int main(int argc, char const *argv[]){
    
    // Declare host (_h) & device (_d)
    unsigned char   *buffer_h, *buffer_d;   //Load the data
    unsigned int    *histo_h, *histo_d;     //result here
    unsigned int    size;
    cudaError_t     err;
    dim3            dim_grid, dim_block;

    // Load info:
    readVector("input.dat", &buffer_h, &size);
    printf("Loaded data of size %d at buffer %p\n", size, &buffer_h);

    // Allocate histagram @ host:
    histo_h =(unsigned int*) malloc(MAX_VAL * sizeof(unsigned int));
    if(histo_h == NULL) exit(1);

    // Allocate device memory
    err = cudaMalloc((void**) &buffer_d, size   * sizeof(unsigned char));
    err = cudaMalloc((void**) &histo_d, MAX_VAL * sizeof(unsigned));

    // Copy host to device & set 0 the result
    cudaMemcpy(buffer_d, buffer_h, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(histo_d, 0, MAX_VAL * sizeof(unsigned int));

    // Launch the histogram kernel
    dim_block   = dim3(BLOCKS);
    dim_grid    = dim3(ceil(size/BLOCKS));
    histo_simple<<<dim_grid, dim_block>>>(buffer_d, size, histo_d);

    // Copy results device to host
    cudaMemcpy(histo_h, histo_d, MAX_VAL * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Launch CPU version
    unsigned histo_g[MAX_VAL];
    memset(histo_g, 0, MAX_VAL * sizeof(unsigned));
    for(int i=0; i<size; i++)
        ++histo_g[buffer_h[i]];

    // Write the results:

    for(int i=0; i < MAX_VAL; i++){
	printf("%u -- expected %u\n", histo_h[i], histo_g[i]);
    }

    writeVectorUnsig("out.dat", histo_h, MAX_VAL);
        
    // Clean host&device memory:
    cudaFree(histo_d);
    cudaFree(buffer_d);
    free(histo_h);
    free(buffer_h);

    return 0;
}
