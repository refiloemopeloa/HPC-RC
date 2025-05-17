#ifndef SPHERE_H
#define SPHERE_H

#include <stdbool.h>
#include <stdio.h>

#include "hypatiaINC.h"
#include "hitRecord.h"

#include "types.h"
#include "ray.h"
#include "material.h"
#include "allocator.h"

typedef struct aabb AABB;

typedef struct {
    // center of the sphere
    vec3 center;

    //material
    Material sphMat;

    // radius of the sphere
    CFLOAT radius;
} Sphere;


__device__ __host__ bool obj_sphereHit(const Sphere* __restrict__  s, Ray r, CFLOAT t_min, CFLOAT t_max, HitRecord * outRecord);
__device__ __host__ bool obj_sphereCalcBoundingBox(const Sphere* __restrict__  s, AABB * outbox);
__device__ __host__ void obj_sphereTexCoords(vec3 pointOnSphere, CFLOAT * outU, CFLOAT * outV);

// enum contaning different types of objects
typedef enum {
    SPHERE,
    OBJLL,
    OBJBVH
} ObjectType;

typedef struct {
    // ptr to the object stored in this node
    void * __restrict__  object;

    // type of the object 
    ObjectType objType;
} Object;


__device__ __host__ Object * obj_createObject(void * __restrict__  object, ObjectType type, 
                                 DynamicStackAlloc * __restrict__  dsa );

// node of the linked list 
typedef struct objectLLNode ObjectLLNode;
typedef struct objectLLNode{
    Object obj;

    // points to the next node
    ObjectLLNode * __restrict__  next;
} ObjectLLNode;

// linked list storing pointer to objects in the scene
typedef struct objectLL {
    // number of objects 
    size_t numObjects;
    
    // points to the first node in the linked list 
    ObjectLLNode * __restrict__  head;
    
    // Dynamic allocation stack
    DynamicStackAlloc * __restrict__  dsa;

    // Linear allocator
    LinearAllocFC * __restrict__  hrAlloc;

    // whether the object is valid or not
    bool valid;
} ObjectLL;


// create and setup an object linked list and return a pointer to it
__device__ __host__ ObjectLL * obj_createObjectLL(
    DynamicStackAlloc * dsaAlloc, 
    DynamicStackAlloc * dsaObjs
);

// function to add an object to the linked list
// returns true if the operation is successful
__device__ __host__ bool obj_objectLLAdd(
        ObjectLL * __restrict__  objll, 
        void * __restrict__  obj, 
        ObjectType objType
);

// function to add spheres
// returns true if operation is successful
__device__ __host__ bool obj_objLLAddSphere(ObjectLL * __restrict__  objll,
        Sphere s);

// remove an object at any index
__device__ __host__ bool obj_objectLLRemove(ObjectLL * __restrict__  objll, size_t index); 

__device__ __host__ Object * obj_objectLLGetAT(const ObjectLL * __restrict__  objll, size_t index);
__device__ __host__ void obj_objectLLSetAT(const ObjectLL * __restrict__  objll, size_t index, Object object);

typedef bool (*ObjectComparator)(const Object * obj1, const Object * obj2);
__device__ __host__ void obj_objectLLSort(const ObjectLL * __restrict__  objll, 
                             size_t start, 
                             size_t end, 
                             ObjectComparator comp);

// returns a hit record if any object in the list is intersected by the given ray
// under the given conditions
// __device__ __host__ /*HitRecord**/bool obj_objLLHit (const ObjectLL* __restrict__  objll, 
//                           Ray r, 
//                           CFLOAT t_min, 
//                           CFLOAT t_max,
//                           HitRecord * out);

__device__ bool obj_objLLHit(const ObjectLL* objll, Ray r, 
                            CFLOAT t_min, CFLOAT t_max, HitRecord* out);

__device__ __host__ bool obj_objectLLCalcBoundingBox(const ObjectLL * __restrict__  objll, AABB* __restrict__  outbox);

typedef struct aabb {
    vec3 maximum;
    vec3 minimum;
} AABB;

__device__ __host__ bool obj_AABBHit(const AABB* __restrict__  s, Ray r, CFLOAT t_min, CFLOAT t_max);


typedef struct {
    AABB box;

    DynamicStackAlloc * __restrict__  dsa;

    Object * __restrict__  right;
    Object * __restrict__  left;
} BVH;

__device__ __host__ BVH * obj_createBVH(DynamicStackAlloc * alloc, DynamicStackAlloc * dsa);

__device__ __host__ void obj_fillBVH(BVH * __restrict__  bvh, 
                          const ObjectLL * __restrict__  objects,
                          size_t start, size_t end); 
__device__ __host__ bool obj_bvhCalcBoundingBox(const BVH * __restrict__  bvh, AABB * __restrict__  outbox);
__device__ __host__ bool obj_bvhHit(const BVH* __restrict__  bvh, Ray r, CFLOAT t_min, CFLOAT t_max, HitRecord * out); 

#endif

