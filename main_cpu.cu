#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <float.h>
#include <omp.h>

#include "include/camera_gpu.cu"

using namespace std;
using namespace std::chrono;

#define OMP

__host__ vec3 color(const Ray &r, hitable **world, uniform_real_distribution<float> &local_rand_state, default_random_engine &generator)
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
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state, generator))
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

#define RND_cpu (local_rand_state(generator))

void create_world_cpu(hitable **h_list, hitable **h_world, camera **h_cam, int x, int y, uniform_real_distribution<float> &local_rand_state, default_random_engine &generator)
{
    h_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
                           new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            float choose_mat = RND_cpu;
            vec3 center(a + RND_cpu, 0.2, b + RND_cpu);
            if (choose_mat < 0.8f)
            {
                h_list[i++] = new sphere(center, 0.2,
                                         new lambertian(vec3(RND_cpu * RND_cpu, RND_cpu * RND_cpu, RND_cpu * RND_cpu)));
            }
            else if (choose_mat < 0.95f)
            {
                h_list[i++] = new sphere(center, 0.2,
                                         new metal(vec3(0.5f * (1.0f + RND_cpu), 0.5f * (1.0f + RND_cpu), 0.5f * (1.0f + RND_cpu)), 0.5f * RND_cpu));
            }
            else
            {
                h_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
            }
        }
    }
    h_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
    h_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    h_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
    *h_world = new hitable_list(h_list, 22 * 22 + 1 + 3);

    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    (lookfrom - lookat).length();
    float aperture = 0.1;
    *h_cam = new camera(lookfrom,
                        lookat,
                        vec3(0, 1, 0),
                        30.0,
                        float(x) / float(y),
                        aperture,
                        dist_to_focus);
}

void free_world_cpu(hitable **h_list, hitable **h_world, camera **h_camera)
{
    for (int i = 0; i < 22 * 22 + 1 + 3; i++)
    {
        delete ((sphere *)h_list[i])->mat_ptr;
        delete h_list[i];
    }
    delete *h_world;
    delete *h_camera;
}

void render_cpu(vec3 *pixels, hitable **world, camera **cam, int x, int y, int s, uniform_real_distribution<float> &local_rand_state, default_random_engine &generator)
{
    float r;
    float g;
    vec3 b;
    int pixel_index;
#ifdef OMP
#pragma omp parallel
    {

        omp_set_num_threads(16);
#pragma omp for collapse(2) private(r, g, b, pixel_index)
#endif
        for (int j = y - 1; j >= 0; j--)
        {
            for (int i = 0; i < x; i++)
            {
                pixel_index = j * x + i;
                vec3 col(0, 0, 0);
                for (int k = 0; k < s; k++)
                {
                    // Random jitter for anti-aliasings
                    float r = float(i + local_rand_state(generator)) / float(x);
                    float g = float(j + local_rand_state(generator)) / float(y);
                    Ray ray = (*cam)->get_ray(r, g, local_rand_state, generator); // Assuming camera is defined globally or passed in some way
                    col += color(ray, world, local_rand_state, generator);
                }
                col /= float(s);
                col[0] = sqrt(col[0]);
                col[1] = sqrt(col[1]);
                col[2] = sqrt(col[2]);
                pixels[pixel_index] = col;
            }
        }
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
    int num_hitables = 22 * 22 + 1 + 3;
    long vec3_bytes = x * y * sizeof(vec3);
    vec3 *h_pixels = (vec3 *)malloc(vec3_bytes);

    vec3 *verify_pixels = (vec3 *)malloc(vec3_bytes);
    hitable **h_list = (hitable **)malloc(num_hitables * sizeof(hitable *));
    hitable **world = (hitable **)malloc(sizeof(hitable *));
    camera **cam = (camera **)malloc(sizeof(camera *));

    create_world_cpu(h_list, world, cam, x, y, distribution, generator);

    cout << "Rendering with CPU..." << endl;
    auto start = high_resolution_clock::now();
    render_cpu(verify_pixels, world, cam, x, y, s, distribution, generator);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<nanoseconds>(stop - start);
    cout << "Time taken by cpu: " << duration.count() / (1e9) << " seconds" << endl;

#ifdef PRINT
    string filename_cpu = "out-Ray-cpu.ppm";
    ofstream file_cpu(filename_cpu);
    print_image(x, y, h_pixels, file_cpu);
    file_cpu.close();
    cout << "Image saved to " << filename_cpu << endl;
#endif
    free_world_cpu(h_list, world, cam);
    free(h_list);
    free(world);
    free(cam);
    free(h_pixels);
    free(verify_pixels);

    return 0;
}