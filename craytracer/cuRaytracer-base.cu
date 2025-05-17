#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

//nvcc cuRaytracer.cu -o cuRaytracer -lcurand
//./cuRaytracer output.ppm


// Define vector/color structure
typedef float CFLOAT;

struct vec3 {
    CFLOAT x, y, z;

    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(CFLOAT x, CFLOAT y, CFLOAT z) : x(x), y(y), z(z) {}

    __host__ __device__ vec3 operator+(const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ vec3 operator-(const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ vec3 operator*(CFLOAT f) const { return vec3(x * f, y * f, z * f); }
    __host__ __device__ CFLOAT dot(const vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ vec3 cross(const vec3 &v) const {
        return vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    __host__ __device__ vec3 normalize() const {
        CFLOAT inv_len = 1.0f / sqrtf(x * x + y * y + z * z);
        return vec3(x * inv_len, y * inv_len, z * inv_len);
    }
};

// Define ray structure
struct Ray {
    vec3 origin;
    vec3 direction;

    __host__ __device__ Ray(const vec3 &o, const vec3 &d) : origin(o), direction(d) {}
    __host__ __device__ vec3 point_at(CFLOAT t) const { return origin + direction * t; }
};

// Define RGB color structures
struct RGBColorF {
    CFLOAT r, g, b;
};

struct RGBColorU8 {
    unsigned char r, g, b;
};

// Define material types
enum MaterialType { LAMBERTIAN, METAL, DIELECTRIC };

// Define material structure
struct Material {
    MaterialType type;
    union {
        struct {
            vec3 albedo;
        } lambertian;
        struct {
            vec3 albedo;
            CFLOAT fuzz;
        } metal;
        struct {
            CFLOAT ir; // index of refraction
        } dielectric;
    };
};

// Define sphere structure
struct Sphere {
    vec3 center;
    CFLOAT radius;
    Material mat;

    __host__ __device__ Sphere(const vec3 &c, CFLOAT r, const Material &m) : center(c), radius(r), mat(m) {}

    __host__ __device__ bool intersect(const Ray &ray, CFLOAT &t) const {
        vec3 oc = ray.origin - center;
        CFLOAT a = ray.direction.dot(ray.direction);
        CFLOAT b = 2.0f * oc.dot(ray.direction);
        CFLOAT c = oc.dot(oc) - radius * radius;
        CFLOAT discriminant = b * b - 4 * a * c;
        
        if (discriminant < 0) return false;
        
        CFLOAT sqrt_discr = sqrtf(discriminant);
        CFLOAT t0 = (-b - sqrt_discr) / (2.0f * a);
        CFLOAT t1 = (-b + sqrt_discr) / (2.0f * a);
        
        t = (t0 < t1) ? t0 : t1;
        return true;
    }
};

// Define camera structure
struct Camera {
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    CFLOAT lens_radius;

    __host__ __device__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, CFLOAT vfov, CFLOAT aspect_ratio, CFLOAT aperture, CFLOAT focus_dist) {
        CFLOAT theta = vfov * M_PI / 180.0f;
        CFLOAT h = tan(theta / 2.0f);
        CFLOAT viewport_height = 2.0f * h;
        CFLOAT viewport_width = aspect_ratio * viewport_height;

        w = (lookfrom - lookat).normalize();
        u = vup.cross(w).normalize();
        v = w.cross(u);

        origin = lookfrom;
        horizontal = u * viewport_width * focus_dist;
        vertical = v * viewport_height * focus_dist;
        lower_left_corner = origin - horizontal * 0.5f - vertical * 0.5f - w * focus_dist;
        lens_radius = aperture / 2.0f;
    }

    __host__ __device__ Ray get_ray(CFLOAT s, CFLOAT t, curandState *local_rand_state) const {
        vec3 rd = random_in_unit_disk(local_rand_state) * lens_radius;
        vec3 offset = u * rd.x + v * rd.y;
        return Ray(origin + offset, 
                  lower_left_corner + horizontal * s + vertical * t - origin - offset);
    }

    __host__ __device__ vec3 random_in_unit_disk(curandState *local_rand_state) const {
        vec3 p;
        do {
            p = vec3(curand_uniform(local_rand_state) * 2.0f - 1.0f,
                     curand_uniform(local_rand_state) * 2.0f - 1.0f,
                     0);
        } while (p.dot(p) >= 1.0f);
        return p;
    }
};

// Random number utilities
__host__ __device__ vec3 random_vec3(curandState *local_rand_state) {
    return vec3(curand_uniform(local_rand_state),
                curand_uniform(local_rand_state),
                curand_uniform(local_rand_state));
}

__host__ __device__ vec3 random_vec3(curandState *local_rand_state, CFLOAT min, CFLOAT max) {
    return vec3(curand_uniform(local_rand_state) * (max - min) + min,
                curand_uniform(local_rand_state) * (max - min) + min,
                curand_uniform(local_rand_state) * (max - min) + min);
}

// Material scattering functions
__host__ __device__ bool scatter_lambertian(const Ray &r_in, const vec3 &normal, const Material &mat, vec3 &attenuation, Ray &scattered, curandState *local_rand_state) {
    vec3 scatter_direction = normal + random_vec3(local_rand_state).normalize();
    if (scatter_direction.near_zero()) scatter_direction = normal;
    scattered = Ray(r_in.point_at(1e-5f), scatter_direction);
    attenuation = mat.lambertian.albedo;
    return true;
}

__host__ __device__ bool scatter_metal(const Ray &r_in, const vec3 &normal, const Material &mat, vec3 &attenuation, Ray &scattered, curandState *local_rand_state) {
    vec3 reflected = reflect(r_in.direction.normalize(), normal);
    scattered = Ray(r_in.point_at(1e-5f), reflected + random_vec3(local_rand_state) * mat.metal.fuzz);
    attenuation = mat.metal.albedo;
    return (scattered.direction.dot(normal) > 0);
}

__host__ __device__ bool scatter_dielectric(const Ray &r_in, const vec3 &normal, const Material &mat, vec3 &attenuation, Ray &scattered, curandState *local_rand_state) {
    attenuation = vec3(1.0f, 1.0f, 1.0f);
    CFLOAT refraction_ratio = mat.dielectric.ir;
    
    vec3 unit_direction = r_in.direction.normalize();
    CFLOAT cos_theta = fminf((-unit_direction).dot(normal), 1.0f);
    CFLOAT sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    
    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    vec3 direction;
    
    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state)) {
        direction = reflect(unit_direction, normal);
    } else {
        direction = refract(unit_direction, normal, refraction_ratio);
    }
    
    scattered = Ray(r_in.point_at(1e-5f), direction);
    return true;
}

__host__ __device__ CFLOAT reflectance(CFLOAT cosine, CFLOAT ref_idx) {
    // Use Schlick's approximation for reflectance
    CFLOAT r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

// Ray color function
__host__ __device__ vec3 ray_color(const Ray &r, Sphere *world, int world_size, int depth, curandState *local_rand_state) {
    if (depth <= 0) {
        return vec3(0.0f, 0.0f, 0.0f);
    }

    CFLOAT closest_t = 1e10f;
    int hit_index = -1;
    CFLOAT temp_t;
    
    // Find closest hit
    for (int i = 0; i < world_size; ++i) {
        if (world[i].intersect(r, temp_t) && temp_t < closest_t) {
            closest_t = temp_t;
            hit_index = i;
        }
    }
    
    if (hit_index >= 0) {
        vec3 hit_point = r.point_at(closest_t);
        vec3 normal = (hit_point - world[hit_index].center).normalize();
        
        Ray scattered;
        vec3 attenuation;
        bool scattered_result = false;
        
        switch (world[hit_index].mat.type) {
            case LAMBERTIAN:
                scattered_result = scatter_lambertian(r, normal, world[hit_index].mat, attenuation, scattered, local_rand_state);
                break;
            case METAL:
                scattered_result = scatter_metal(r, normal, world[hit_index].mat, attenuation, scattered, local_rand_state);
                break;
            case DIELECTRIC:
                scattered_result = scatter_dielectric(r, normal, world[hit_index].mat, attenuation, scattered, local_rand_state);
                break;
        }
        
        if (scattered_result) {
            return attenuation * ray_color(scattered, world, world_size, depth - 1, local_rand_state);
        }
        return vec3(0.0f, 0.0f, 0.0f);
    }
    
    // Background
    vec3 unit_direction = r.direction.normalize();
    CFLOAT t = 0.5f * (unit_direction.y + 1.0f);
    return vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + vec3(0.5f, 0.7f, 1.0f) * t;
}

// Kernel function to render the image
__global__ void render_kernel(RGBColorU8 *fb, int width, int height, int samples_per_pixel, int max_depth, Sphere *world, int world_size, Camera camera) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= width || j >= height) return;
    
    curandState local_rand_state;
    curand_init(1984 + i + j * width, 0, 0, &local_rand_state);
    
    vec3 pixel_color(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < samples_per_pixel; ++s) {
        CFLOAT u = (CFLOAT(i) + curand_uniform(&local_rand_state)) / (width - 1);
        CFLOAT v = (CFLOAT(j) + curand_uniform(&local_rand_state)) / (height - 1);
        Ray r = camera.get_ray(u, v, &local_rand_state);
        pixel_color = pixel_color + ray_color(r, world, world_size, max_depth, &local_rand_state);
    }
    
    // Apply gamma correction and write to framebuffer
    CFLOAT scale = 1.0f / samples_per_pixel;
    pixel_color.x = sqrtf(pixel_color.x * scale);
    pixel_color.y = sqrtf(pixel_color.y * scale);
    pixel_color.z = sqrtf(pixel_color.z * scale);
    
    int pixel_index = j * width + i;
    fb[pixel_index].r = (unsigned char)(fminf(1.0f, pixel_color.x) * 255);
    fb[pixel_index].g = (unsigned char)(fminf(1.0f, pixel_color.y) * 255);
    fb[pixel_index].b = (unsigned char)(fminf(1.0f, pixel_color.z) * 255);
}

// Function to create a random scene
void random_scene(Sphere *d_world, int &world_size) {
    Sphere *world = new Sphere[500];
    world_size = 0;
    
    // Ground
    Material ground_material;
    ground_material.type = LAMBERTIAN;
    ground_material.lambertian.albedo = vec3(0.5f, 0.5f, 0.5f);
    world[world_size++] = Sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material);
    
    // Random small spheres
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            CFLOAT choose_mat = rand() / (CFLOAT)RAND_MAX;
            vec3 center(a + 0.9f * rand() / (CFLOAT)RAND_MAX, 0.2f, b + 0.9f * rand() / (CFLOAT)RAND_MAX);
            
            if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                Material sphere_material;
                
                if (choose_mat < 0.8f) {
                    // Diffuse
                    vec3 albedo = vec3(rand() / (CFLOAT)RAND_MAX, rand() / (CFLOAT)RAND_MAX, rand() / (CFLOAT)RAND_MAX) * 
                                 vec3(rand() / (CFLOAT)RAND_MAX, rand() / (CFLOAT)RAND_MAX, rand() / (CFLOAT)RAND_MAX);
                    sphere_material.type = LAMBERTIAN;
                    sphere_material.lambertian.albedo = albedo;
                    world[world_size++] = Sphere(center, 0.2f, sphere_material);
                } else if (choose_mat < 0.95f) {
                    // Metal
                    vec3 albedo = vec3(0.5f + 0.5f * rand() / (CFLOAT)RAND_MAX, 
                                      0.5f + 0.5f * rand() / (CFLOAT)RAND_MAX, 
                                      0.5f + 0.5f * rand() / (CFLOAT)RAND_MAX);
                    CFLOAT fuzz = 0.5f * rand() / (CFLOAT)RAND_MAX;
                    sphere_material.type = METAL;
                    sphere_material.metal.albedo = albedo;
                    sphere_material.metal.fuzz = fuzz;
                    world[world_size++] = Sphere(center, 0.2f, sphere_material);
                } else {
                    // Glass
                    sphere_material.type = DIELECTRIC;
                    sphere_material.dielectric.ir = 1.5f;
                    world[world_size++] = Sphere(center, 0.2f, sphere_material);
                }
            }
        }
    }
    
    // Three large spheres
    Material material1;
    material1.type = DIELECTRIC;
    material1.dielectric.ir = 1.5f;
    world[world_size++] = Sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, material1);
    
    Material material2;
    material2.type = LAMBERTIAN;
    material2.lambertian.albedo = vec3(0.4f, 0.2f, 0.1f);
    world[world_size++] = Sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, material2);
    
    Material material3;
    material3.type = METAL;
    material3.metal.albedo = vec3(0.7f, 0.6f, 0.5f);
    material3.metal.fuzz = 0.0f;
    world[world_size++] = Sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, material3);
    
    // Copy to device
    cudaMemcpy(d_world, world, world_size * sizeof(Sphere), cudaMemcpyHostToDevice);
    delete[] world;
}

// Save image to PPM file
void save_image(const char *filename, RGBColorU8 *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            RGBColorU8 pixel = image[y * width + x];
            fwrite(&pixel.r, 1, 1, fp);
            fwrite(&pixel.g, 1, 1, fp);
            fwrite(&pixel.b, 1, 1, fp);
        }
    }
    
    fclose(fp);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("FATAL ERROR: Output file name not provided.\n");
        printf("EXITING ...\n");
        return 0;
    }

    // Image dimensions and settings
    const CFLOAT aspect_ratio = 16.0f / 9.0f;
    const int WIDTH = 640;
    const int HEIGHT = 640;
    const int SAMPLES_PER_PIXEL = 100;
    const int MAX_DEPTH = 50;
    
    // Allocate frame buffer on host and device
    RGBColorU8 *h_fb = (RGBColorU8*)malloc(WIDTH * HEIGHT * sizeof(RGBColorU8));
    RGBColorU8 *d_fb;
    cudaMalloc(&d_fb, WIDTH * HEIGHT * sizeof(RGBColorU8));
    
    // Create scene on device
    Sphere *d_world;
    int world_size;
    cudaMalloc(&d_world, 500 * sizeof(Sphere));
    random_scene(d_world, world_size);
    
    // Set up camera
    vec3 lookfrom(13.0f, 2.0f, 3.0f);
    vec3 lookat(0.0f, 0.0f, 0.0f);
    vec3 vup(0.0f, 1.0f, 0.0f);
    CFLOAT dist_to_focus = 10.0f;
    CFLOAT aperture = 0.1f;
    Camera camera(lookfrom, lookat, vup, 20.0f, aspect_ratio, aperture, dist_to_focus);
    
    // Launch kernel
    dim3 block_size(16, 16);
    dim3 grid_size((WIDTH + block_size.x - 1) / block_size.x, 
                   (HEIGHT + block_size.y - 1) / block_size.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    render_kernel<<<grid_size, block_size>>>(d_fb, WIDTH, HEIGHT, SAMPLES_PER_PIXEL, MAX_DEPTH, d_world, world_size, camera);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Render time: %.2f ms\n", milliseconds);
    
    // Copy frame buffer back to host
    cudaMemcpy(h_fb, d_fb, WIDTH * HEIGHT * sizeof(RGBColorU8), cudaMemcpyDeviceToHost);
    
    // Save image
    save_image(argv[1], h_fb, WIDTH, HEIGHT);
    
    // Cleanup
    free(h_fb);
    cudaFree(d_fb);
    cudaFree(d_world);
    
    printf("Raytracing completed. Image saved to %s\n", argv[1]);
    return 0;
}