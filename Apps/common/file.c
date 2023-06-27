#include <stdlib.h>
#include <stdio.h>

#include "file.h"

void readVector(const char *fName, float **vec_h, unsigned *size)
{
    FILE* fp = fopen(fName, "rb");

    if (fp == NULL) FATAL("Cannot open input file");

    fread(size, sizeof(unsigned), 1, fp);
    *vec_h = (float*)malloc(*size * sizeof(float));
    if(*vec_h == NULL) FATAL("Unable to allocate host");
    fread(*vec_h, sizeof(float), *size, fp);
    fclose(fp);
}


void writeVector(const char *fName, float *vec_h, unsigned size)
{
    FILE* fp = fopen(fName, "wb");
    if (fp == NULL) FATAL("Cannot open output file");
    fwrite(&size, sizeof(unsigned), 1, fp);
    fwrite(vec_h, sizeof(float), size, fp);
    fclose(fp);
}