#ifndef TEXTURE_H
#define TEXTURE_H

#include <stdint.h>


#include "hypatiaINC.h"
#include "color.h"
#include "types.h"

typedef enum {
    SOLID_COLOR,
    CHECKER,
    IMAGE
} TexType;

typedef struct {
    void * __restrict__  tex;
    TexType texType;
} Texture;

typedef struct {
    RGBColorF color;
} SolidColor;


typedef struct {
    Texture odd;
    Texture even;
} Checker;


typedef struct {
    uint8_t * data;
    int32_t width;
    int32_t height;
    uint32_t bytesPerScanLine;
    int32_t compsPerPixel;
} Image;

__device__ __host__ RGBColorF tex_value(const Texture * __restrict__  t, 
        CFLOAT u, CFLOAT v, vec3 p);

__device__ __host__ RGBColorF tex_solidColorValue(const SolidColor * __restrict__  t);
__device__ __host__ RGBColorF tex_checkerValue(const Checker * __restrict__  c,
                CFLOAT u, CFLOAT v, vec3 p);

__device__ __host__ void tex_loadImage(Image * __restrict__  img, const char* filename);
__device__ __host__ RGBColorF tex_imageValue(const Image * __restrict__  img, 
                CFLOAT u, CFLOAT v);

#endif
