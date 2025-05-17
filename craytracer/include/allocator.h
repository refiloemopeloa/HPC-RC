#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef NDEBUG
typedef struct debugStruct {
    size_t allocatedChunks;
} DebugStruct;
#endif

typedef struct poolAllocNode PoolAllocNode;
typedef struct poolAllocNode{
    PoolAllocNode * __restrict__ next; 
} PoolAllocNode;

typedef struct poolalloc PoolAlloc;
// Pool allocator
typedef struct poolalloc {

    // size of the memory in this allocator
    size_t totalSize;

    // size of each chunk
    size_t chunkSize;
    
    // information for debugging
#ifndef NDEBUG
    DebugStruct dbgS;
#endif 
    // points to the first node in the free list
    PoolAllocNode * __restrict__ head;

    // the pointer to the buffer held by this allocator
    uint8_t * __restrict__ buffptr;
} PoolAlloc;



// function to create a pool allocator
__device__ __host__ PoolAlloc* alloc_createPoolAllocator(size_t size, size_t chunkAlignment, size_t chunkSize);

// function to allocate chunks
__device__ __host__ void * alloc_poolAllocAllocate(PoolAlloc * __restrict__ pa);

// free all the allocated chunk in allocator
// doesn't deallocate the memory allocated by the createPoolAllocator function
__device__ __host__ void alloc_poolAllocFreeAll(PoolAlloc * __restrict__ pa); 

// free the chunk given by ptr 
// ptr should be allocate by using poolAllocAllocate function
__device__ __host__ void alloc_poolAllocFree(PoolAlloc * __restrict__ pa, void * __restrict__ ptr);

// free pool allocator
__device__ __host__ void alloc_freePoolAllocator(PoolAlloc * __restrict__ pAlloc);


/*
 *
 *
 * Static linear allocator that can allocate only one chunk size of the alignment
 *
 * */
typedef struct linearAllocatorFC {
    uint8_t * bufptr;
    size_t totalSize;
    size_t curOffset; 
    size_t chunkSize;
    size_t alignment;
} LinearAllocFC;


// create a linearAllocator
__device__ __host__ LinearAllocFC * alloc_createLinearAllocFC(size_t numChunks, 
                                                 size_t chunkSize, 
                                                 size_t chunkAlignment);

// allocate memory
__device__ void * alloc_linearAllocFCAllocate(LinearAllocFC * __restrict__ lafc);

// free all
__device__ __host__ void alloc_linearAllocFCFreeAll(LinearAllocFC * __restrict__ lafc);

// destroy the linear allocator
__device__ __host__ void alloc_freeLinearAllocFC(LinearAllocFC * __restrict__ lafc);

/* 
 *
 * Static stack allocator
 *
 * */

typedef struct stackAllocHeader{
    size_t padding;
    size_t prevOffset;
} StackAllocHeader;

// Static stack allocator
typedef struct stackAlloc {
    bool isFull;

    // pointer to the buffer
    uint8_t * buffptr;    

    // size of the buffer
    size_t totalSize;

    // offset of the memory that will be 
    // allocated next
    size_t offset;

    // value of offset before previous 
    // allocation 
    size_t prevOffset;
} StackAlloc;

// create stack allocator
__device__ __host__ StackAlloc* alloc_createStackAllocator(size_t size);

// allocate memory 
__device__ __host__ void * alloc_stackAllocAllocate(StackAlloc * __restrict__ sa, size_t allocSize, size_t alignment);

// free the most recent allocation
__device__ __host__ bool alloc_stackAllocFree(StackAlloc * __restrict__ sa, void * ptr);

// free all the allocations
__device__ __host__ void alloc_stackAllocFreeAll(StackAlloc * __restrict__ sa);

// destroy the stack allocator
__device__ __host__ void alloc_freeStackAllocator(StackAlloc * sa);

/* 
 *
 * Dynamic stack allocator
 *
 * */

typedef struct ptrStack {
    size_t curOffset; 
    size_t maxPointers;
    void ** bufptr;
    bool valid;
} PtrStack;

__device__ __host__ void alloc_createPtrStack(PtrStack * __restrict__ ps, size_t maxPointers);
__device__ __host__ bool alloc_ptrStackPush(PtrStack * __restrict__ ps, void * val);
__device__ __host__ bool alloc_ptrStackPop(PtrStack * __restrict__ ps, void ** __restrict__ out);
__device__ __host__ void alloc_freePtrStack(PtrStack * __restrict__ ptr);

typedef struct dynamicStackAlloc {
    PtrStack ps;
    size_t allocatorSize;

    size_t numAllocatedStacks;

    // is the allocator full 
    bool isMax;
    bool valid;
} DynamicStackAlloc;

__device__ __host__ DynamicStackAlloc * alloc_createDynamicStackAllocD(
        size_t maxAllocatorSize,
        size_t maxAllocators);


__device__ __host__ void alloc_createDynamicStackAlloc(
        DynamicStackAlloc * __restrict__ dsa, 
        size_t maxAllocatorSize,
        size_t maxAllocators);

__device__ __host__ void* alloc_dynamicStackAllocAllocate(
        DynamicStackAlloc * __restrict__ dsa,
        size_t allocSize,
        size_t alignment);

__device__ __host__ bool alloc_dynamicStackAllocFree(DynamicStackAlloc * __restrict__ dsa, void * ptr);
__device__ __host__ bool alloc_dynamicStackAllocFreeAll(DynamicStackAlloc * __restrict__ dsa);
__device__ __host__ void alloc_freeDynamicStackAlloc(DynamicStackAlloc * __restrict__ dsa);
__device__ __host__ void alloc_freeDynamicStackAllocD(DynamicStackAlloc * __restrict__ dsa);

#endif

