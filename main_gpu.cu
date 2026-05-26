#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <curand_kernel.h>
#include <omp.h>

#include "include/camera_gpu.cu"

using namespace std;
using namespace std::chrono;

#define USE_MEMORY 0 // shared=1, constant=2, none=0

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Shared memory structure for sphere data
struct sphere_data
{
    vec3 center;
    float radius;
    int material_type; // 0=lambertian, 1=metal, 2=dielectric
    vec3 albedo;
    float fuzz_or_ref_idx;
};

// Compact camera data for shared memory
struct camera_data
{
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

#define RND (curand_uniform(local_rand_state))

// Constant memory declarations
#if USE_MEMORY == 2
#define MAX_SPHERES 500 // Adjust based on your scene complexity
__constant__ sphere_data const_spheres[MAX_SPHERES];
__constant__ camera_data const_camera;
__constant__ int const_num_spheres;

__device__ vec3 color_constant_memory(const Ray &r, curandState *local_rand_state)
{
    Ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < 50; depth++)
    {
        float closest_t = FLT_MAX;
        int hit_sphere_idx = -1;
        vec3 hit_point, hit_normal;

        // Check intersection with all spheres using constant memory
        for (int i = 0; i < const_num_spheres; i++)
        {
            const sphere_data &sphere = const_spheres[i];
            vec3 oc = cur_ray.origin() - sphere.center;
            float a = dot(cur_ray.direction(), cur_ray.direction());
            float b = 2.0f * dot(oc, cur_ray.direction());
            float c = dot(oc, oc) - sphere.radius * sphere.radius;
            float discriminant = b * b - 4 * a * c;

            if (discriminant > 0)
            {
                float t = (-b - sqrt(discriminant)) / (2.0f * a);
                if (t > 0.001f && t < closest_t)
                {
                    closest_t = t;
                    hit_sphere_idx = i;
                    hit_point = cur_ray.point_at_parameter(t);
                    hit_normal = (hit_point - sphere.center) / sphere.radius;
                }
            }
        }

        if (hit_sphere_idx >= 0)
        {
            const sphere_data &hit_sphere = const_spheres[hit_sphere_idx];

            // Material scattering logic
            vec3 attenuation;
            vec3 scattered_direction;
            bool scattered = false;

            if (hit_sphere.material_type == 0) // Lambertian
            {
                vec3 target = hit_point + hit_normal;
                // Add random point in unit sphere
                vec3 random_vec;
                do
                {
                    random_vec = 2.0f * vec3(RND, RND, RND) - vec3(1.0f, 1.0f, 1.0f);
                } while (dot(random_vec, random_vec) >= 1.0f);

                target += random_vec;
                scattered_direction = target - hit_point;
                attenuation = hit_sphere.albedo;
                scattered = true;
            }
            else if (hit_sphere.material_type == 1) // Metal
            {
                vec3 reflected = reflect(unit_vector(cur_ray.direction()), hit_normal);

                // Add fuzziness
                vec3 random_vec;
                do
                {
                    random_vec = 2.0f * vec3(RND, RND, RND) - vec3(1.0f, 1.0f, 1.0f);
                } while (dot(random_vec, random_vec) >= 1.0f);

                scattered_direction = reflected + hit_sphere.fuzz_or_ref_idx * random_vec;
                attenuation = hit_sphere.albedo;
                scattered = dot(scattered_direction, hit_normal) > 0;
            }
            else if (hit_sphere.material_type == 2) // Dielectric
            {
                attenuation = vec3(1.0f, 1.0f, 1.0f);
                float ref_idx = hit_sphere.fuzz_or_ref_idx;

                vec3 outward_normal;
                float ni_over_nt;
                float cosine;

                if (dot(cur_ray.direction(), hit_normal) > 0)
                {
                    outward_normal = -hit_normal;
                    ni_over_nt = ref_idx;
                    cosine = ref_idx * dot(cur_ray.direction(), hit_normal) / cur_ray.direction().length();
                }
                else
                {
                    outward_normal = hit_normal;
                    ni_over_nt = 1.0f / ref_idx;
                    cosine = -dot(cur_ray.direction(), hit_normal) / cur_ray.direction().length();
                }

                vec3 refracted;
                if (refract(cur_ray.direction(), outward_normal, ni_over_nt, refracted))
                {
                    float reflect_prob = schlick(cosine, ref_idx);
                    if (RND < reflect_prob)
                    {
                        scattered_direction = reflect(cur_ray.direction(), hit_normal);
                    }
                    else
                    {
                        scattered_direction = refracted;
                    }
                }
                else
                {
                    scattered_direction = reflect(cur_ray.direction(), hit_normal);
                }
                scattered = true;
            }

            if (scattered)
            {
                cur_attenuation *= attenuation;
                cur_ray = Ray(hit_point, scattered_direction);
            }
            else
            {
                return vec3(0, 0, 0);
            }
        }
        else
        {
            // Sky gradient
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }
    return vec3(0, 0, 0);
}

__global__ void render_constant_memory(vec3 *pixels, int x, int y, int s, curandState *rand_state)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i >= x) || (j >= y))
        return;

    int pixel_index = j * x + i;
    curandState local_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);

    for (int k = 0; k < s; k++)
    {
        float u = float(i + curand_uniform(&local_state)) / float(x);
        float v = float(j + curand_uniform(&local_state)) / float(y);

        // Generate ray using constant camera data
        vec3 rd = const_camera.lens_radius * random_in_unit_disk(&local_state);
        vec3 offset = const_camera.u * rd.x() + const_camera.v * rd.y();

        Ray ray(const_camera.origin + offset,
                const_camera.lower_left_corner + u * const_camera.horizontal + v * const_camera.vertical - const_camera.origin - offset);

        col += color_constant_memory(ray, &local_state);
    }

    rand_state[pixel_index] = local_state;
    col /= float(s);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    pixels[pixel_index] = col;
}

// Function to copy world data to constant memory
void copy_to_constant_memory(hitable **world, camera **cam, int num_spheres)
{
    // Host buffers for staging data
    sphere_data *h_sphere_data = new sphere_data[num_spheres];
    camera_data h_camera_data;

    // Copy camera data to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(const_camera, &h_camera_data, sizeof(camera_data)));

    // Copy sphere data to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(const_spheres, h_sphere_data, num_spheres * sizeof(sphere_data)));

    // Copy number of spheres
    checkCudaErrors(cudaMemcpyToSymbol(const_num_spheres, &num_spheres, sizeof(int)));

    delete[] h_sphere_data;
}

// Kernel to extract world data for constant memory
__global__ void extract_world_data_for_constant(hitable **world, sphere_data *sphere_buffer,
                                                camera **cam, camera_data *cam_buffer, int num_spheres)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Extract camera data
        camera *c = *cam;
        cam_buffer->origin = c->origin;
        cam_buffer->lower_left_corner = c->lower_left_corner;
        cam_buffer->horizontal = c->horizontal;
        cam_buffer->vertical = c->vertical;
        cam_buffer->u = c->u;
        cam_buffer->v = c->v;
        cam_buffer->w = c->w;
        cam_buffer->lens_radius = c->lens_radius;

        // Extract sphere data
        hitable_list *world_list = (hitable_list *)*world;
        for (int i = 0; i < num_spheres && i < world_list->list_size; i++)
        {
            sphere *s = (sphere *)world_list->list[i];
            sphere_buffer[i].center = s->center;
            sphere_buffer[i].radius = s->radius;

            int mat_type = s->mat_ptr->get_type();
            sphere_buffer[i].material_type = mat_type;

            if (mat_type == 0) // lambertian
            {
                lambertian *lamb = (lambertian *)s->mat_ptr;
                sphere_buffer[i].albedo = lamb->albedo;
                sphere_buffer[i].fuzz_or_ref_idx = 0.0f;
            }
            else if (mat_type == 1) // metal
            {
                metal *met = (metal *)s->mat_ptr;
                sphere_buffer[i].albedo = met->albedo;
                sphere_buffer[i].fuzz_or_ref_idx = met->fuzz;
            }
            else if (mat_type == 2) // dielectric
            {
                dielectric *diel = (dielectric *)s->mat_ptr;
                sphere_buffer[i].albedo = vec3(1.0f, 1.0f, 1.0f);
                sphere_buffer[i].fuzz_or_ref_idx = diel->ref_idx;
            }
        }
    }
}

// Main rendering function with constant memory optimization
void render_with_constant_memory(dim3 threads_per_block, dim3 blocks_per_grid, vec3 *d_pixels, int x, int y, int s,
                                 hitable **d_world, camera **d_cam,
                                 curandState *d_rand_state, int num_spheres,
                                 time_point<std::chrono::high_resolution_clock> &start_time,
                                 time_point<std::chrono::high_resolution_clock> &end_time)
{
    // Allocate temporary buffers for data extraction
    sphere_data *d_sphere_buffer, *h_sphere_buffer;
    camera_data *d_camera_buffer, h_camera_buffer;

    checkCudaErrors(cudaMalloc((void **)&d_sphere_buffer, num_spheres * sizeof(sphere_data)));
    checkCudaErrors(cudaMalloc((void **)&d_camera_buffer, sizeof(camera_data)));
    h_sphere_buffer = new sphere_data[num_spheres];

    // Extract data from world structures
    extract_world_data_for_constant<<<1, 1>>>(d_world, d_sphere_buffer, d_cam, d_camera_buffer, num_spheres);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy data to host
    checkCudaErrors(cudaMemcpy(h_sphere_buffer, d_sphere_buffer, num_spheres * sizeof(sphere_data), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_camera_buffer, d_camera_buffer, sizeof(camera_data), cudaMemcpyDeviceToHost));

    // Copy to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(const_spheres, h_sphere_buffer, num_spheres * sizeof(sphere_data)));
    checkCudaErrors(cudaMemcpyToSymbol(const_camera, &h_camera_buffer, sizeof(camera_data)));
    checkCudaErrors(cudaMemcpyToSymbol(const_num_spheres, &num_spheres, sizeof(int)));

    start_time = high_resolution_clock::now();
    render_constant_memory<<<blocks_per_grid, threads_per_block>>>(d_pixels, x, y, s, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    end_time = high_resolution_clock::now();

    // Cleanup
    checkCudaErrors(cudaFree(d_sphere_buffer));
    checkCudaErrors(cudaFree(d_camera_buffer));
    delete[] h_sphere_buffer;
}
#elif USE_MEMORY == 1

__device__ vec3 color_optimized(const Ray &r, sphere_data *shared_spheres, int num_spheres, curandState *local_rand_state)
{
    Ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < 50; depth++)
    {
        float closest_t = FLT_MAX;
        int hit_sphere_idx = -1;
        vec3 hit_point, hit_normal;

        // Check intersection with all spheres using shared memory
        for (int i = 0; i < num_spheres; i++)
        {
            sphere_data &sphere = shared_spheres[i];
            vec3 oc = cur_ray.origin() - sphere.center;
            float a = dot(cur_ray.direction(), cur_ray.direction());
            float b = 2.0f * dot(oc, cur_ray.direction());
            float c = dot(oc, oc) - sphere.radius * sphere.radius;
            float discriminant = b * b - 4 * a * c;

            if (discriminant > 0)
            {
                float t = (-b - sqrt(discriminant)) / (2.0f * a);
                if (t > 0.001f && t < closest_t)
                {
                    closest_t = t;
                    hit_sphere_idx = i;
                    hit_point = cur_ray.point_at_parameter(t);
                    hit_normal = (hit_point - sphere.center) / sphere.radius;
                }
            }
        }

        if (hit_sphere_idx >= 0)
        {
            sphere_data &hit_sphere = shared_spheres[hit_sphere_idx];

            // Material scattering logic
            vec3 attenuation;
            vec3 scattered_direction;
            bool scattered = false;

            if (hit_sphere.material_type == 0) // Lambertian
            {
                // Lambertian scattering
                vec3 target = hit_point + hit_normal;
                // Add random point in unit sphere
                vec3 random_vec;
                do
                {
                    random_vec = 2.0f * vec3(RND, RND, RND) - vec3(1.0f, 1.0f, 1.0f);
                } while (dot(random_vec, random_vec) >= 1.0f);

                target += random_vec;
                scattered_direction = target - hit_point;
                attenuation = hit_sphere.albedo;
                scattered = true;
            }
            else if (hit_sphere.material_type == 1) // Metal
            {
                vec3 reflected = reflect(unit_vector(cur_ray.direction()), hit_normal);

                // Add fuzziness
                vec3 random_vec;
                do
                {
                    random_vec = 2.0f * vec3(RND, RND, RND) - vec3(1.0f, 1.0f, 1.0f);
                } while (dot(random_vec, random_vec) >= 1.0f);

                scattered_direction = reflected + hit_sphere.fuzz_or_ref_idx * random_vec;
                attenuation = hit_sphere.albedo;
                scattered = dot(scattered_direction, hit_normal) > 0;
            }
            else if (hit_sphere.material_type == 2) // Dielectric
            {
                attenuation = vec3(1.0f, 1.0f, 1.0f);
                float ref_idx = hit_sphere.fuzz_or_ref_idx;

                vec3 outward_normal;
                float ni_over_nt;
                float cosine;

                if (dot(cur_ray.direction(), hit_normal) > 0)
                {
                    outward_normal = -hit_normal;
                    ni_over_nt = ref_idx;
                    cosine = ref_idx * dot(cur_ray.direction(), hit_normal) / cur_ray.direction().length();
                }
                else
                {
                    outward_normal = hit_normal;
                    ni_over_nt = 1.0f / ref_idx;
                    cosine = -dot(cur_ray.direction(), hit_normal) / cur_ray.direction().length();
                }

                vec3 refracted;
                if (refract(cur_ray.direction(), outward_normal, ni_over_nt, refracted))
                {
                    float reflect_prob = schlick(cosine, ref_idx);
                    if (RND < reflect_prob)
                    {
                        scattered_direction = reflect(cur_ray.direction(), hit_normal);
                    }
                    else
                    {
                        scattered_direction = refracted;
                    }
                }
                else
                {
                    scattered_direction = reflect(cur_ray.direction(), hit_normal);
                }
                scattered = true;
            }

            if (scattered)
            {
                cur_attenuation *= attenuation;
                cur_ray = Ray(hit_point, scattered_direction);
            }
            else
            {
                return vec3(0, 0, 0);
            }
        }
        else
        {
            // Sky gradient
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }
    return vec3(0, 0, 0);
}

__global__ void render_optimized(vec3 *pixels, int x, int y, int s,
                                 camera_data cam_data, sphere_data *world_spheres,
                                 int num_spheres, curandState *rand_state)
{
    // Allocate shared memory for spheres
    extern __shared__ sphere_data shared_spheres[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int threads_per_block = blockDim.x * blockDim.y;

    // Cooperatively load sphere data into shared memory
    for (int i = tid; i < num_spheres; i += threads_per_block)
    {
        shared_spheres[i] = world_spheres[i];
    }

    __syncthreads(); // Wait for all threads to finish loading

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i >= x) || (j >= y))
        return;

    int pixel_index = j * x + i;
    curandState local_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);

    for (int k = 0; k < s; k++)
    {
        float u = float(i + curand_uniform(&local_state)) / float(x);
        float v = float(j + curand_uniform(&local_state)) / float(y);

        // Generate ray using camera data (fixed implementation)
        vec3 rd = cam_data.lens_radius * random_in_unit_disk(&local_state);
        vec3 offset = cam_data.u * rd.x() + cam_data.v * rd.y();

        vec3 ray_origin = cam_data.origin + offset;
        vec3 ray_direction = cam_data.lower_left_corner + u * cam_data.horizontal + v * cam_data.vertical - cam_data.origin - offset;

        Ray ray(ray_origin, ray_direction);

        col += color_optimized(ray, shared_spheres, num_spheres, &local_state);
    }

    rand_state[pixel_index] = local_state;
    col /= float(s);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    pixels[pixel_index] = col;
}

#else USE_MEMORY == 0
__device__ vec3 color(const Ray &r, hitable **world, curandState *local_rand_state)
{
    Ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f); // Initial attenuation set to white

    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            Ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation; // Update attenuation
                cur_ray = scattered;            // Update ray to scattered ray
            }
            else
            {
                return vec3(0, 0, 0); // Return black if no scattering
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f); // Gradient from white to blue
            return cur_attenuation * c;
        }
    }
    return vec3(0, 0, 0); // Return black if max iterations reached
}

__global__ void render(vec3 *pixels, int x, int y, int s, camera **cam, hitable **world, curandState *rand_state)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i >= x) || (j >= y))
        return;
    {
        int pixel_index = j * x + i;
        curandState local_state = rand_state[pixel_index];
        vec3 col(0, 0, 0);
        for (int k = 0; k < s; k++)
        {
            // Random jitter for anti-aliasings
            float r = float(i + curand_uniform(&local_state)) / float(x);
            float g = float(j + curand_uniform(&local_state)) / float(y);
            Ray ray = (*cam)->get_ray(r, g, &local_state); // Assuming camera is defined globally or passed in some way
            col += color(ray, world, &local_state);
        }

        rand_state[pixel_index] = local_state; // Save the state back to global memory
        col /= float(s);
        col[0] = sqrt(col[0]);
        col[1] = sqrt(col[1]);
        col[2] = sqrt(col[2]);
        pixels[pixel_index] = col;
    }
}
#endif

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_cam, int x, int y, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));

        int i = 1;

        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f)
                {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f)
                {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else
                {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_cam = new camera(lookfrom,
                            lookat,
                            vec3(0, 1, 0),
                            30.0,
                            float(x) / float(y),
                            aperture,
                            dist_to_focus);
    }
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int x, int y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= x) || (j >= y))
        return;
    int pixel_index = j * x + i;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

#if USE_MEMORY == 1
__global__ void extract_world_data(hitable **world, sphere_data *sphere_buffer,
                                   camera **cam, camera_data *cam_buffer)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Extract camera data
        camera *c = *cam;
        cam_buffer->origin = c->origin;
        cam_buffer->lower_left_corner = c->lower_left_corner;
        cam_buffer->horizontal = c->horizontal;
        cam_buffer->vertical = c->vertical;
        cam_buffer->u = c->u;
        cam_buffer->v = c->v;
        cam_buffer->w = c->w;
        cam_buffer->lens_radius = c->lens_radius;

        // Extract sphere data
        hitable_list *world_list = (hitable_list *)*world;
        for (int i = 0; i < world_list->list_size; i++)
        {
            sphere *s = (sphere *)world_list->list[i];
            sphere_buffer[i].center = s->center;
            sphere_buffer[i].radius = s->radius;

            // Get material type and properties
            int mat_type = s->mat_ptr->get_type();
            sphere_buffer[i].material_type = mat_type;

            if (mat_type == 0) // lambertian
            {
                lambertian *lamb = (lambertian *)s->mat_ptr;
                sphere_buffer[i].albedo = lamb->albedo;
                sphere_buffer[i].fuzz_or_ref_idx = 0.0f;
            }
            else if (mat_type == 1) // metal
            {
                metal *met = (metal *)s->mat_ptr;
                sphere_buffer[i].albedo = met->albedo;
                sphere_buffer[i].fuzz_or_ref_idx = met->fuzz;
            }
            else if (mat_type == 2) // dielectric
            {
                dielectric *diel = (dielectric *)s->mat_ptr;
                sphere_buffer[i].albedo = vec3(1.0f, 1.0f, 1.0f);
                sphere_buffer[i].fuzz_or_ref_idx = diel->ref_idx;
            }
        }
    }
}

#endif

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_cam)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < 22 * 22 + 1 + 3; i++)
        {
            delete ((sphere *)d_list[i])->mat_ptr;
            delete d_list[i];
        }
        delete *d_world;
        delete *d_cam;
    }
}

void print_image(int x, int y, vec3 *pixels, ostream &file)
{
    file << "P3\n"
         << x << " " << y << "\n255\n";

    for (int j = y - 1; j >= 0; j--)
    {
        for (int i = 0; i < x; i++)
        {
            int pixel_index = j * x + i;
            file << int(255.99 * pixels[pixel_index].r()) << " " << int(255.99 * pixels[pixel_index].g()) << " " << int(255.99 * pixels[pixel_index].b()) << "\n";
        }
    }
}

int main(int argc, char *argv[])
{

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, 1.0);

    int x, y, s;

    if (argc < 2 || argc > 3)
    {
        x = 200;
        s = 10;
    }
    else if (argc == 2)
    {
        x = atoi(argv[1]);
        s = 10;
    }
    else if (argc == 3)
    {
        x = atoi(argv[1]);
        s = atoi(argv[2]);
    }

    y = x / 2;
    int n = x * y;
    long bytes = n * sizeof(int);
    long vec3_bytes = x * y * sizeof(vec3);

    vec3 *h_pixels = (vec3 *)malloc(vec3_bytes);

    vec3 *d_pixels;
    checkCudaErrors(cudaMalloc((void **)&d_pixels, vec3_bytes));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, n * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hitable **d_list;
    int num_hitables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));

    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));

    camera **d_cam;
    checkCudaErrors(cudaMalloc((void **)&d_cam, sizeof(camera *)));

    create_world<<<1, 1>>>(d_list, d_world, d_cam, x, y, d_rand_state2);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((x + threads_per_block.x - 1) / threads_per_block.x,
                         (y + threads_per_block.y - 1) / threads_per_block.y);

    int num_spheres = 22 * 22 + 1 + 3;

#if USE_MEMORY == 2
    cout << "Rendering with GPU (constant memory optimization)..." << endl;
    time_point<std::chrono::high_resolution_clock> start;
    time_point<std::chrono::high_resolution_clock> stop;

    render_with_constant_memory(threads_per_block, blocks_per_grid, d_pixels, x, y, s, d_world, d_cam, d_rand_state, num_spheres, start, stop);

#elif USE_MEMORY == 1
    sphere_data *d_sphere_data;
    camera_data *d_camera_data;

    checkCudaErrors(cudaMalloc((void **)&d_sphere_data, num_spheres * sizeof(sphere_data)));
    checkCudaErrors(cudaMalloc((void **)&d_camera_data, sizeof(camera_data)));
    extract_world_data<<<1, 1>>>(d_world, d_sphere_data, d_cam, d_camera_data);
    checkCudaErrors(cudaDeviceSynchronize());
    // Copy camera data to host for kernel launch
    camera_data h_camera_data;
    checkCudaErrors(cudaMemcpy(&h_camera_data, d_camera_data, sizeof(camera_data), cudaMemcpyDeviceToHost));

    // Calculate shared memory size needed
    size_t shared_mem_size = num_spheres * sizeof(sphere_data);
    cout << "Rendering with GPU (optimized with shared memory)..." << endl;
    auto start = high_resolution_clock::now();

    render_optimized<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_pixels, x, y, s, h_camera_data, d_sphere_data, num_spheres, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto stop = high_resolution_clock::now();

#elif USE_MEMORY == 0
    // size_t shared_mem_size = (threads_per_block.x * threads_per_block.y * sizeof(curandState)) + sizeof(camera_params);

    render_init<<<blocks_per_grid, threads_per_block>>>(x, y, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cout << "Rendering with GPU..." << endl;
    auto start = high_resolution_clock::now();
    render<<<blocks_per_grid, threads_per_block>>>(d_pixels, x, y, s, d_cam, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto stop = high_resolution_clock::now();
#endif
    checkCudaErrors(cudaMemcpy(h_pixels, d_pixels, vec3_bytes, cudaMemcpyDeviceToHost));
    auto duration = duration_cast<nanoseconds>(stop - start);
    cout << "Time taken by gpu: " << duration.count() / 1e9 << " seconds" << endl;


#ifdef PRINT
    string filename_gpu = "out-Ray-gpu.ppm";
    ofstream file_gpu(filename_gpu);
    print_image(x, y, h_pixels, file_gpu);
    file_gpu.close();
    cout << "Image saved to " << filename_gpu << endl;
#endif

    // Free memory

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_cam);
    checkCudaErrors(cudaGetLastError());
#ifdef CONSTANT_MEMORY
    checkCudaErrors(cudaFree(d_sphere_data));
    checkCudaErrors(cudaFree(d_camera_data));
#else
#ifdef SHARED_MEMORY
    checkCudaErrors(cudaFree(d_sphere_data));
    checkCudaErrors(cudaFree(d_camera_data));
#endif
#endif
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_pixels));
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(d_rand_state));

    cudaDeviceReset();

    free(h_pixels);

    return 0;
}
