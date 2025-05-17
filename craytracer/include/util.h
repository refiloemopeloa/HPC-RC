#ifndef UTIL_H
#define UTIL_H

#include "hypatiaINC.h"
#include "types.h"

#include <stdint.h>
#include <stdbool.h>

__device__ __host__ CFLOAT util_floatClamp(CFLOAT c, CFLOAT lower, CFLOAT upper);
__device__ __host__ uint8_t util_uint8Clamp(uint8_t c, uint8_t lower, uint8_t upper);

// not in use
__device__ __host__ uint32_t util_randomRange(uint32_t lower, uint32_t upper);


__device__ __host__ vec3 util_randomUnitSphere();

__device__ __host__ vec3 util_randomUnitVector();
__device__ __host__ CFLOAT util_randomFloat(CFLOAT lower, CFLOAT upper);

__device__ __host__ vec3 util_vec3Reflect(vec3 v,vec3 n);

__device__ __host__ bool util_isVec3Zero(vec3 v);

__device__ __host__ vec3 util_randomUnitDisk();

#endif

