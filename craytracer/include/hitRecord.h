#ifndef HITRECORD_H
#define HITRECORD_H

#include <stdbool.h>

#include "hypatiaINC.h"
#include "types.h"

typedef struct material Material;

typedef struct {
    // intersection point of ray and sphere
    vec3 point;

    // normal at the point of intersection
    vec3 normal;

    // distance between origin and point
    CFLOAT distanceFromOrigin;

    // pointer to the material of the object
    const Material * hitObjMat;    

    // texture coordinates
    CFLOAT u;
    CFLOAT v;

    // the surface of the object where ray
    // intersected the object
    bool frontFace;

    // true when this structure is valid
    bool valid;
} HitRecord;

// construct a hit record object and return it
__device__ __host__ HitRecord hr_setRecord(
            CFLOAT distanceFromOrigin, 
            vec3 point, 
            vec3 normal, 
            vec3 direction, 
            const Material * __restrict__  hitObjMat
        );
__device__ __host__ void hr_setRecordi(
            CFLOAT distanceFromOrigin, 
            vec3 point, 
            vec3 normal, 
            vec3 direction, HitRecord* __restrict__  outRecord,
            const Material * __restrict__  hitObjMat
        );

#endif

