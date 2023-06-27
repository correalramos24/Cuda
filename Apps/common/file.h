#ifndef __FILEH__
#define __FILEH__

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#ifdef __cplusplus
    extern "C" {
#endif


void readVector (const char* fName, float **vec_h, unsigned *size);
void writeVectorFloat(const char* fName, float *vec_h, unsigned size);
void writeVectorUnsig(const char* fName, unsigned *vec, unsigned size);

#ifdef __cplusplus
    }
#endif


#endif
