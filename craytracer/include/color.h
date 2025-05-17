#ifndef COLOR_H
#define COLOR_H

#include <stdint.h>
#include "types.h"

typedef struct rgbColorU8 {

    uint8_t r;

    uint8_t g;

    uint8_t b;

} RGBColorU8;

typedef struct rgbColorF {

    CFLOAT r;

    CFLOAT g;

    CFLOAT b;

} RGBColorF;




__device__ __host__ RGBColorU8 coloru8_create(uint8_t r, uint8_t g, uint8_t b);
// a
// __device__ __host__ RGBColorU8 coloru8_createf(CFLOAT r, CFLOAT g, CFLOAT b);
__device__ RGBColorU8 coloru8_createf(CFLOAT r, CFLOAT g, CFLOAT b);

__device__ __host__ RGBColorF colorf_create(CFLOAT r, CFLOAT g, CFLOAT b);
__device__ __host__ RGBColorF convertU8toF(RGBColorU8 in);

__device__ __host__ RGBColorU8 convertFtoU8(RGBColorF in);

// a
// __device__ __host__ RGBColorF colorf_multiply(RGBColorF x, RGBColorF y);
// __device__ __host__ RGBColorF colorf_add(RGBColorF x, RGBColorF y);
__device__ RGBColorF colorf_multiply(RGBColorF a, RGBColorF b);
__device__ RGBColorF colorf_add(RGBColorF a, RGBColorF b);

// Replace _Generic with a simple version
#define COLOR_U8CREATE(r, g, b) coloru8_createf((r), (g), (b))

#endif