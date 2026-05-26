#include "vec3_gpu.cu"

class Ray {
    public:
        __host__ __device__ Ray() {}
        __host__ __device__ Ray(const vec3& a, const vec3& b) { A = a; B = b; }
        __host__ __device__ vec3 origin() const { return A; }
        __host__ __device__ vec3 direction() const { return B; }
        __host__ __device__ vec3 point_at_parameter(float t) const {
            return A + B*t;
        }

        vec3 A;
        vec3 B;
};