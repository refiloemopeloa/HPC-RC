#include "ray_gpu.cu"

#define HITABLE
#ifdef HITABLE

class material; // Forward declaration

struct hit_record {
    float t;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
};

class hitable {
public:
    __host__ __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif

