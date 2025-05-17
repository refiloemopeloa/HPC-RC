#ifndef OUTFILE_H
#define OUTFILE_H

#include "hypatiaINC.h"
#include "color.h"

__device__ __host__ void writeToPPM(const char * filename, int width, int height, const RGBColorU8* arr);

#endif

