__global__ 
void histo_simple(  unsigned char *buffer, 
                    long size, 
                    unsigned *histo);

__global__ 
void histo_privatization(   unsigned char *buffer, 
                            long size, 
                            unsigned *histo);