#define HYPATIA_IMPLEMENTATION
// #define STB_IMAGE_IMPLEMENTATION

#include <assert.h>
#include <stdalign.h>
#include <stdbool.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include "helper_functions.h" // For PGM image loading/saving
#include "helper_cuda.h"

#include "allocator.h"
#include "camera.h"
#include "color.h"
#include "hitRecord.h"
#include "hypatiaINC.h"
#include "material.h"
#include "outfile.h"
#include "ray.h"
#include "sphere.h"
#include "texture.h"
#include "types.h"
#include "util.h"

#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCudaError(T err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (int)err, cudaGetErrorString(err), func);
        exit(1);
    }
}

__device__ void vector3_normalize_device(vec3* v) {
    CFLOAT length = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);
    if (length > 0) {
        v->x /= length;
        v->y /= length;
        v->z /= length;
    }
}

// // Add this implementation to your cudaMain.cu file
// __device__ bool obj_objLLHit(const ObjectLL* objll, Ray r, CFLOAT t_min, CFLOAT t_max, HitRecord* out) {
//     // Make sure the ObjectLL is valid
//     if (!objll || !objll->valid) {
//         return false;
//     }

//     bool hit_anything = false;
//     CFLOAT closest_so_far = t_max;
//     HitRecord temp_rec;

//     // Iterate through the linked list of objects
//     ObjectLLNode* current = objll->head;
//     while (current != NULL) {
//         bool hit = false;
        
//         // Check hit based on object type
//         switch (current->obj.objType) {
//             case SPHERE:
//                 hit = obj_sphereHit((Sphere*)current->obj.object, r, t_min, closest_so_far, &temp_rec);
//                 break;
//             case OBJLL:
//                 hit = obj_objLLHit((ObjectLL*)current->obj.object, r, t_min, closest_so_far, &temp_rec);
//                 break;
//             case OBJBVH:
//                 // Assuming you have a BVH hit function
//                 hit = obj_bvhHit((BVH*)current->obj.object, r, t_min, closest_so_far, &temp_rec);
//                 break;
//             default:
//                 hit = false;
//         }

//         if (hit) {
//             hit_anything = true;
//             closest_so_far = temp_rec.distanceFromOrigin;
//             *out = temp_rec;
//         }

//         current = current->next;
//     }

//     return hit_anything;
// }

// Device-side random number generator states
__device__ curandState *devStates;

__device__ RGBColorU8 writeColor(CFLOAT r, CFLOAT g, CFLOAT b, int sample_per_pixel) {
    CFLOAT scale = 1.0 / sample_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);
    
    // Direct implementation instead of macro
    return (RGBColorU8){
        .r = (uint8_t)(255.999 * r),
        .g = (uint8_t)(255.999 * g),
        .b = (uint8_t)(255.999 * b)
    };
}

// Iterative version of ray_c for CUDA
__device__ RGBColorF cudaRayColor(Ray r, ObjectLL *world, int max_depth, curandState *local_state)
{
    RGBColorF accumulated_color = {0};
    RGBColorF attenuation = {1.0, 1.0, 1.0};

    for (int depth = 0; depth < max_depth; depth++)
    {
        HitRecord rec;
        rec.valid = false;

        if (!obj_objLLHit(world, r, 0.00001f, FLT_MAX, &rec))
        {
            // Background
            vec3 unit_direction = r.direction;
            vector3_normalize_device(&unit_direction);
            CFLOAT t = 0.5f * (unit_direction.y + 1.0f);
            vec3 color_vec = {
                1.0f - t + t * 0.5f,
                1.0f - t + t * 0.7f,
                1.0f - t + t * 1.0f};
            RGBColorF bg_color = {color_vec.x, color_vec.y, color_vec.z};
            return colorf_multiply(accumulated_color, bg_color);
        }

        Ray scattered;
        RGBColorF albedo;
        if (!mat_scatter(&r, &rec, &albedo, &scattered))
        {
            return accumulated_color;
        }

        attenuation = colorf_multiply(attenuation, albedo);
        r = scattered;
        accumulated_color = colorf_add(accumulated_color, attenuation);
    }

    return accumulated_color;
}

__global__ void renderKernel(RGBColorU8 *image, Camera camera, ObjectLL *world,
                             int width, int height, int samples, int max_depth,
                             curandState *states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height)
        return;

    int pixel_index = j * width + i;
    curandState local_state = states[pixel_index];

    CFLOAT u, v;
    RGBColorF pixel_color = {0};

    for (int s = 0; s < samples; s++)
    {
        u = (i + curand_uniform(&local_state)) / (width - 1);
        v = (j + curand_uniform(&local_state)) / (height - 1);

        Ray r = cam_getRay(&camera, u, v);
        pixel_color = colorf_add(pixel_color, cudaRayColor(r, world, max_depth, &local_state));
    }

    image[pixel_index] = writeColor(pixel_color.r, pixel_color.g, pixel_color.b, samples);
    states[pixel_index] = local_state;
}

// Kernel for RNG initialization
__global__ void setupRNG(curandState *state, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

void printProgressBar(int i, int max)
{
    int p = (int)(100 * (CFLOAT)i / max);

    printf("|");
    for (int j = 0; j < p; j++)
    {
        printf("=");
    }

    for (int j = p; j < 100; j++)
    {
        printf("*");
    }

    if (p == 100)
    {
        printf("|\n");
    }
    else
    {
        printf("|\r");
    }
}

#define randomFloat() util_randomFloat(0.0, 1.0)

void randomSpheres(ObjectLL *world, DynamicStackAlloc *dsa)
{

    LambertianMat *materialGround = (LambertianMat *)alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));
    SolidColor *sc1 = (SolidColor *)alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                                    alignof(SolidColor));

    SolidColor *sc = (SolidColor *)alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                                   alignof(SolidColor));

    Checker *c =
        (Checker *)alloc_dynamicStackAllocAllocate(dsa, sizeof(Checker), alignof(Checker));

    // sc1->color = (RGBColorF){.r = 0.2, .b = 0.3, .g = 0.1};
    sc1->color.r = 0.2;
    sc1->color.g = 0.1;
    sc1->color.b = 0.3;
    sc->color = (RGBColorF){.r = 0.9, .g = 0.9, .b = 0.9};

    c->even.tex = sc1;
    c->even.texType = SOLID_COLOR;
    c->odd.tex = sc;
    c->odd.texType = SOLID_COLOR;

    materialGround->lambTexture.tex = c;
    materialGround->lambTexture.texType = CHECKER;

    /*materialGround->albedo.r = 0.5;
    materialGround->albedo.g = 0.5;
    materialGround->albedo.b = 0.5;*/

    // obj_objLLAddSphere(world,
    //                    (Sphere){.center = {.x = 0, .y = -1000, .z = 0},
    //                             .radius = 1000,
    //                             .sphMat = MAT_CREATE_LAMB_IP(materialGround)});

    Sphere s;
    s.center.x = 0;
    s.center.y = -1000;
    s.center.z = 0;
    s.radius = 1000;
    s.sphMat = MAT_CREATE_LAMB_IP(materialGround);
    obj_objLLAddSphere(world, s);

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            CFLOAT chooseMat = randomFloat();
            vec3 center;
            center.x = a + 0.9 * randomFloat();
            center.y = 0.2;
            center.z = b + 0.9 * randomFloat();

            CFLOAT length = sqrtf((center.x - 4) * (center.x - 4) +
                                  (center.y - 0.2) * (center.y - 0.2) +
                                  (center.z - 0) * (center.z - 0));

            if (length > 0.9)
            {
                if (chooseMat < 0.8)
                {
                    // diffuse
                    RGBColorF albedo = {
                        .r = randomFloat() * randomFloat(),
                        .g = randomFloat() * randomFloat(),
                        .b = randomFloat() * randomFloat(),
                    };

                    LambertianMat *lambMat = (LambertianMat *)alloc_dynamicStackAllocAllocate(
                        dsa, sizeof(LambertianMat), alignof(LambertianMat));

                    SolidColor *sc = (SolidColor *)alloc_dynamicStackAllocAllocate(
                        dsa, sizeof(SolidColor), alignof(SolidColor));

                    sc->color = albedo;

                    lambMat->lambTexture.tex = sc;
                    lambMat->lambTexture.texType = SOLID_COLOR;

                    // obj_objLLAddSphere(
                    //     world, (Sphere){.center = center,
                    //                     .radius = 0.2,
                    //                     .sphMat = MAT_CREATE_LAMB_IP(lambMat)});

                    s.center = center;
                    s.radius = 0.2;
                    s.sphMat = MAT_CREATE_LAMB_IP(lambMat);
                    obj_objLLAddSphere(world, s);
                }
                else if (chooseMat < 0.95)
                {
                    // metal
                    RGBColorF albedo = {.r = util_randomFloat(0.5, 1.0),
                                        .g = util_randomFloat(0.5, 1.0),
                                        .b = util_randomFloat(0.5, 1.0)};
                    CFLOAT fuzz = util_randomFloat(0.5, 1.0);

                    MetalMat *metalMat = (MetalMat *)alloc_dynamicStackAllocAllocate(
                        dsa, sizeof(MetalMat), alignof(MetalMat));

                    metalMat->albedo = albedo;
                    metalMat->fuzz = fuzz;

                    // obj_objLLAddSphere(
                    //     world,
                    //     (Sphere){.center = center,
                    //              .radius = 0.2,
                    //              .sphMat = MAT_CREATE_METAL_IP(metalMat)});

                    Sphere s;
                    s.center = center;
                    s.radius = 0.2;
                    s.sphMat = MAT_CREATE_METAL_IP(metalMat);
                    obj_objLLAddSphere(world, s);
                }
                else
                {
                    DielectricMat *dMat = (DielectricMat *)alloc_dynamicStackAllocAllocate(
                        dsa, sizeof(DielectricMat), alignof(DielectricMat));
                    dMat->ir = 1.5;
                    // obj_objLLAddSphere(
                    //     world,
                    //     (Sphere){.center = center,
                    //              .radius = 0.2,
                    //              .sphMat = MAT_CREATE_DIELECTRIC_IP(dMat)});

                    s.center = center;
                    s.radius = 0.2;
                    s.sphMat = MAT_CREATE_DIELECTRIC_IP(dMat);
                    obj_objLLAddSphere(world, s);
                }
            }
        }
    }

    DielectricMat *material1 = (DielectricMat *)alloc_dynamicStackAllocAllocate(
        dsa, sizeof(DielectricMat), alignof(DielectricMat));
    material1->ir = 1.5;

    // obj_objLLAddSphere(world,
    //                    (Sphere){.center = {.x = 0, .y = 1, .z = 0},
    //                             .radius = 1.0,
    //                             .sphMat = MAT_CREATE_DIELECTRIC_IP(material1)});

    s.center.x = 0;
    s.center.y = 1;
    s.center.z = 0;
    s.radius = 1.0;
    s.sphMat = MAT_CREATE_DIELECTRIC_IP(material1);
    obj_objLLAddSphere(world, s);

    LambertianMat *material2 = (LambertianMat *)alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));

    sc = (SolidColor *)alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                       alignof(SolidColor));

    sc->color = (RGBColorF){.r = 0.4, .g = 0.2, .b = 0.1};
    material2->lambTexture.tex = sc;
    material2->lambTexture.texType = SOLID_COLOR;
    /*material2->albedo.r = 0.4;
    material2->albedo.g = 0.2;
    material2->albedo.b = 0.1;
    */

    // obj_objLLAddSphere(world,
    //                    (Sphere){.center = {.x = -4, .y = 1, .z = 0},
    //                             .radius = 1.0,
    //                             .sphMat = MAT_CREATE_LAMB_IP(material2)});

    s.center.x = -4;
    s.center.y = 1;
    s.center.z = 0;
    s.radius = 1.0;
    s.sphMat = MAT_CREATE_LAMB_IP(materialGround);
    obj_objLLAddSphere(world, s);

    MetalMat *material3 = (MetalMat *)alloc_dynamicStackAllocAllocate(dsa, sizeof(MetalMat),
                                                                      alignof(MetalMat));
    material3->albedo.r = 0.7;
    material3->albedo.g = 0.6;
    material3->albedo.b = 0.5;
    material3->fuzz = 0.0;

    // obj_objLLAddSphere(world,
    //                    (Sphere){.center = {.x = 4, .y = 1, .z = 0},
    //                             .radius = 1.0,
    //                             .sphMat = MAT_CREATE_METAL_IP(material3)});

    s.center.x = 4;
    s.center.y = 1;
    s.center.z = 0;
    s.radius = 1.0;
    s.sphMat = MAT_CREATE_METAL_IP(material3);
    obj_objLLAddSphere(world, s);
}

CFLOAT lcg(int *n)
{

    static int seed;
    const int m = 2147483647;
    const int a = 1103515245;
    const int c = 12345;

    if (n != NULL)
    {
        seed = *n;
    }

    seed = (seed * a + c) % m;
    *n = seed;

    return fabs((CFLOAT)seed / m);
}

void randomSpheres2(ObjectLL *world, DynamicStackAlloc *dsa, int n,
                    Image *imgs, int *seed)
{

    LambertianMat *materialGround = (LambertianMat *)alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));
    SolidColor *sc1 = (SolidColor *)alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                                    alignof(SolidColor));

    SolidColor *sc = (SolidColor *)alloc_dynamicStackAllocAllocate(dsa, sizeof(SolidColor),
                                                                   alignof(SolidColor));

    Checker *c =
        (Checker *)alloc_dynamicStackAllocAllocate(dsa, sizeof(Checker), alignof(Checker));

    sc1->color = (RGBColorF){.r = 0.0, .g = 0.0, .b = 0.0};
    sc->color = (RGBColorF){.r = 0.4, .g = 0.4, .b = 0.4};

    c->even.tex = sc1;
    c->even.texType = SOLID_COLOR;
    c->odd.tex = sc;
    c->odd.texType = SOLID_COLOR;

    materialGround->lambTexture.tex = c;
    materialGround->lambTexture.texType = CHECKER;

    /*materialGround->albedo.r = 0.5;
    materialGround->albedo.g = 0.5;
    materialGround->albedo.b = 0.5;*/

    // obj_objLLAddSphere(world,
    //                    (Sphere){.center = {.x = 0, .y = -1000, .z = 0},
    //                             .radius = 1000,
    //                             .sphMat = MAT_CREATE_LAMB_IP(materialGround)});

    Sphere s;
    s.center.x = 0;
    s.center.y = -1000;
    s.center.z = 0;
    s.radius = 1000;
    s.sphMat = MAT_CREATE_LAMB_IP(materialGround);
    obj_objLLAddSphere(world, s);

    for (int a = -2; a < 9; a++)
    {
        for (int b = -9; b < 9; b++)
        {
            CFLOAT chooseMat = lcg(seed);
            // vec3 center = {
            //     .x = a + 0.9 * lcg(seed), .y = 0.2, .z = b + 0.9 * lcg(seed)};

            vec3 center;
            center.x = a + 0.9 * lcg(seed);
            center.y = 0.2;
            center.z = b + 0.9 * lcg(seed);

            if (chooseMat < 0.8)
            {
                // diffuse
                RGBColorF albedo = {
                    .r = lcg(seed) * lcg(seed),
                    .g = lcg(seed) * lcg(seed),
                    .b = lcg(seed) * lcg(seed),

                };

                LambertianMat *lambMat = (LambertianMat *)alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(LambertianMat), alignof(LambertianMat));

                SolidColor *sc = (SolidColor *)alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(SolidColor), alignof(SolidColor));

                sc->color = albedo;

                lambMat->lambTexture.tex = sc;
                lambMat->lambTexture.texType = SOLID_COLOR;

                // obj_objLLAddSphere(
                //     world, (Sphere){.center = center,
                //                     .radius = 0.2,
                //                     .sphMat = MAT_CREATE_LAMB_IP(lambMat)});

                s.center = center;
                s.radius = 0.2;
                s.sphMat = MAT_CREATE_LAMB_IP(lambMat);
                obj_objLLAddSphere(world, s);
            }
            else if (chooseMat < 0.95)
            {
                // metal
                RGBColorF albedo = {.r = lcg(seed) / 2 + 0.5,
                                    .g = lcg(seed) / 2 + 0.5,
                                    .b = lcg(seed) / 2 + 0.5};
                CFLOAT fuzz = lcg(seed) / 2 + 0.5;

                MetalMat *metalMat = (MetalMat *)alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(MetalMat), alignof(MetalMat));

                metalMat->albedo = albedo;
                metalMat->fuzz = fuzz;

                // obj_objLLAddSphere(
                //     world, (Sphere){.center = center,
                //                     .radius = 0.2,
                //                     .sphMat = MAT_CREATE_METAL_IP(metalMat)});

                s.center = center;
                s.radius = 0.2;
                s.sphMat = MAT_CREATE_METAL_IP(metalMat);
                obj_objLLAddSphere(world, s);
            }
            else
            {
                DielectricMat *dMat = (DielectricMat *) alloc_dynamicStackAllocAllocate(
                    dsa, sizeof(DielectricMat), alignof(DielectricMat));
                dMat->ir = 1.5;
                // obj_objLLAddSphere(
                //     world, (Sphere){.center = center,
                //                     .radius = 0.2,
                //                     .sphMat = MAT_CREATE_DIELECTRIC_IP(dMat)});

                s.center = center;
                s.radius = 0.2;
                s.sphMat = MAT_CREATE_DIELECTRIC_IP(dMat);
                obj_objLLAddSphere(world, s);
                
            }
        }
    }

    LambertianMat *material2 = (LambertianMat *) alloc_dynamicStackAllocAllocate(
        dsa, sizeof(LambertianMat), alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[0];
    material2->lambTexture.texType = IMAGE;

    // obj_objLLAddSphere(world,
    //                    (Sphere){.center = {.x = -4, .y = 1, .z = 0},
    //                             .radius = 1.0,
    //                             .sphMat = MAT_CREATE_LAMB_IP(material2)});

    s.center.x = -4;
    s.center.y = 1;
    s.center.z = 0;
    s.radius = 1.0;
    s.sphMat = MAT_CREATE_LAMB_IP(material2);
    obj_objLLAddSphere(world, s);

    material2 = (LambertianMat *) alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[1];
    material2->lambTexture.texType = IMAGE;

    // obj_objLLAddSphere(world,
    //                    (Sphere){.center = {.x = -4, .y = 1, .z = -2.2},
    //                             .radius = 1.0,
    //                             .sphMat = MAT_CREATE_LAMB_IP(material2)});

    s.center.x = -4;
    s.center.y = 1;
    s.center.z = -2.2;
    s.radius = 1.0;
    s.sphMat = MAT_CREATE_LAMB_IP(material2);
    obj_objLLAddSphere(world, s);

    material2 = (LambertianMat *) alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[2];
    material2->lambTexture.texType = IMAGE;

    // obj_objLLAddSphere(world,
    //                    (Sphere){.center = {.x = -4, .y = 1, .z = +2.2},
    //                             .radius = 1.0,
    //                             .sphMat = MAT_CREATE_LAMB_IP(material2)});

    s.center.x = -4;
    s.center.y = 1;
    s.center.z = 2.2;
    s.radius = 1.0;
    s.sphMat = MAT_CREATE_LAMB_IP(material2);
    obj_objLLAddSphere(world, s);

    material2 = (LambertianMat *) alloc_dynamicStackAllocAllocate(dsa, sizeof(LambertianMat),
                                                alignof(LambertianMat));

    material2->lambTexture.tex = &imgs[3];
    material2->lambTexture.texType = IMAGE;

    // obj_objLLAddSphere(world,
    //                    (Sphere){.center = {.x = -4, .y = 1, .z = -4.2},
    //                             .radius = 1.0,
    //                             .sphMat = MAT_CREATE_LAMB_IP(material2)});
    //                                 Sphere s;
    s.center.x = -4;
    s.center.y = 1;
    s.center.z = -4.2;
    s.radius = 1.0;
    s.sphMat = MAT_CREATE_LAMB_IP(material2);
    obj_objLLAddSphere(world, s);
}
#undef randomFloat

// Main CUDA render function
void cudaRender(RGBColorU8 *image, int width, int height, int samples,
                int max_depth, Camera *camera, ObjectLL *world)
{
    // Allocate device memory
    RGBColorU8 *d_image;
    Camera *d_camera;
    ObjectLL *d_world;

    CHECK_CUDA_ERROR(cudaMalloc(&d_image, width * height * sizeof(RGBColorU8)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_camera, sizeof(Camera)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_world, sizeof(ObjectLL)));

    // Allocate RNG states
    curandState *d_states;
    CHECK_CUDA_ERROR(cudaMalloc(&d_states, width * height * sizeof(curandState)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice));
    // Note: You'll need to properly copy the ObjectLL structure and its contents

    // Setup RNG
    dim3 rngBlocks(16, 16);
    dim3 rngGrid((width + rngBlocks.x - 1) / rngBlocks.x,
                 (height + rngBlocks.y - 1) / rngBlocks.y);
    setupRNG<<<rngGrid, rngBlocks>>>(d_states, time(NULL));

    // Render
    dim3 blocks(16, 16);
    dim3 grid((width + blocks.x - 1) / blocks.x,
              (height + blocks.y - 1) / blocks.y);

    renderKernel<<<grid, blocks>>>(d_image, *d_camera, d_world,
                                   width, height, samples, max_depth, d_states);

    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(image, d_image, width * height * sizeof(RGBColorU8), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_camera);
    cudaFree(d_world);
    cudaFree(d_states);
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        printf("FATAL ERROR: Output file name not provided.\n");
        printf("EXITING ...\n");
        return 0;
    }

    srand(time(NULL));

    //    const CFLOAT aspect_ratio = 16.0 / 9.0;
    //    const int WIDTH = 1024;
    //    const int HEIGHT = (int)(WIDTH / aspect_ratio);
    const CFLOAT aspect_ratio = 16.0 / 9.0;
    const int WIDTH = 640;
    const int HEIGHT = 640;
    const int SAMPLES_PER_PIXEL = 100;
    const int MAX_DEPTH = 50;
    RGBColorU8 *image =
        (RGBColorU8 *)malloc(sizeof(RGBColorF) * HEIGHT * WIDTH);

    CFLOAT start = omp_get_wtime();

    uint32_t stepSize = 500;
    uint32_t totalSteps = (WIDTH * HEIGHT) / stepSize + 1;
    size_t stepsCompleted = 0;

    // Initialize CUDA
    cudaFree(0); // Warm-up CUDA runtime

    // Set up scene (same as before)
    DynamicStackAlloc *dsa = alloc_createDynamicStackAllocD(1024, 100);
    DynamicStackAlloc *dsa0 = alloc_createDynamicStackAllocD(1024, 10);
    ObjectLL *world = obj_createObjectLL(dsa0, dsa);

    Image img[4];
    tex_loadImage(&img[0], "./test_textures/kitchen_probe.jpg");
    tex_loadImage(&img[1], "./test_textures/campus_probe.jpg");
    tex_loadImage(&img[2], "./test_textures/building_probe.jpg");
    tex_loadImage(&img[3], "./test_textures/kitchen_probe.jpg");

    int seed = 100;
    randomSpheres2(world, dsa, 4, img, &seed);

    // Set up camera
    vec3 lookFrom = {13.0, 2.0, 3.0};
    vec3 lookAt = {0.0, 0.0, 0.0};
    vec3 up = {0.0, 1.0, 0.0};
    Camera camera;
    cam_setLookAtCamera(&camera, lookFrom, lookAt, up, 20, aspect_ratio, 0.1, 10.0);

    // Allocate image buffer
    image = (RGBColorU8 *)malloc(sizeof(RGBColorU8) * HEIGHT * WIDTH);

    // Render with CUDA
    start = omp_get_wtime();
    cudaRender(image, WIDTH, HEIGHT, SAMPLES_PER_PIXEL, MAX_DEPTH, &camera, world);
    CFLOAT end = omp_get_wtime();

    printf("CUDA Render Time: %lf seconds\n", end - start);

    // Save image and cleanup
    writeToPPM(argv[1], WIDTH, HEIGHT, image);
    free(image);
    alloc_freeDynamicStackAllocD(dsa);
    alloc_freeDynamicStackAllocD(dsa0);

    return 0;
}