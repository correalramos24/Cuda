#ifndef KERNEL_H
#define KERNEL_H

__global__ void vecAdd( float *C, 
                        const float * __restrict__ A, 
                        const float * __restrict__ B, 
                        const unsigned size);

#endif