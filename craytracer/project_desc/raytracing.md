# Ray Tracing

Ray tracing is a long-standing computer graphics technique used to generate realistic images by simulating the way rays of 
light interact with objects in a scene. It traces the path of light as it travels through pixels in an image 
plane, calculating color, shadows, reflections, and refractions to create highly detailed and accurate visual effects. 
Based on the physical phemonemon of light transportation, ray tracing is effectively simulating how light would interact with
the world. 

Unsurprisingly, such a technique is computationally intensive, as it requires simulating numerous light rays interacting with 
complex scenes, involving many recursive calculations for different lighting phemonemon across millions of pixels.

Fortunately, ray tracing is highly parallelisable, as each ray can be processed independently—allowing thousands or even 
millions of rays to be computed simultaneously across multiple threads, significantly accelerating rendering 
performance.

Following the simple ray tracing example in Lec8 lecture slides, we would like to develop a more realistic 
raytracer using CUDA. As a starting point, a raytracer has been provided [here](https://github.com/kaustubh0201/Ray-Tracer/blob/main/).
The code is written in C/C++ with OpenMP parallelisation. 

## Brief Explanantion of RT
Let's go through some important snippets of code for a better understanding of how ratracing works.
For a more detailed explanation of ray tracing, please go [here](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-ray-tracing/how-does-it-work.html).

### Initialisation
Firstly, like any other rendering problem, we need to define the world and the location of the camera from which we are looking
at the world:

Firstly, the a primitive camera requires three set of base parameters: *location*, *direction* and *up*. *Location* and *direction*
are self-explanatory. *Up* simply specifies which direction in the world we consider as the direction that is pointing up (we need
this to make sure our render does not come out up-side-down).
```
// main.c

int main (int argc, char *argv[]){

    //...

    // set camera parameters
    vec3 lookFrom = {.x = 13.0, .y = 2.0, .z = 3.0}; // The camera location
    vec3 lookAt = {.x = 0.0, .y = 0.0, .z = 0.0}; // The camera direction
    vec3 up = {.x = 0.0, .y = 1.0, .z = 0.0}; // Specify which direction is "Up". 

    CFLOAT distToFocus = 10.0; // Focal length
    CFLOAT aperture = 0.1; // Aperture size

    Camera c; // The camera object
    cam_setLookAtCamera(&c, lookFrom, lookAt, up, 20, aspect_ratio,
                        aperture, distToFocus);
```                        

Then we setup the world. For simplicity, we only consider opaque spheres, which can be defined simply mathematically.
```
    //...

    // memory allocation for the objects in the world
    DynamicStackAlloc *dsa = alloc_createDynamicStackAllocD(1024, 100);
    DynamicStackAlloc *dsa0 = alloc_createDynamicStackAllocD(1024, 10);
    ObjectLL *world = obj_createObjectLL(dsa0, dsa); // The world is a linked list of objects

    // loading the textures
    Image img[4];
    tex_loadImage(&img[0], "./images/tex1.jpg");
    tex_loadImage(&img[1], "./images/tex2.jpg");
    tex_loadImage(&img[2], "./images/tex3.jpg");
    tex_loadImage(&img[3], "./images/tex4.jpg");


    // Procedurally add spheres into the world 
    randomSpheres2(world, dsa, 4, img, &seed);
```

### Raytracing

The most important and computationally intensive method in this technique is the recursive tracing of rays. The idea is 
simple: we want to simulate how light rays would interact with objects in the world (spheres in this case). However, as one light
ray would hit the surface of an object, a new ray would be reflected at the point of intersection. We need to consider as many
rays as possible in order to render realistic images. For more information on raytracing, please refer to 

For each pixel in the rendering image, we cast a ray from the camera through the image plane into the scene (as shown in 
\ref{fig:raytracing} and trace the ray to compute the colour and material information of all the objects it hits along the way.
Because every ray interact with the scene completely independent of every other ray, we can parallelise this step. Using `#pragma omp for`,
we can assign different threads to trace rays from different pixels independently.

![Ray Tracing Diagram](./ray-tracing-image-1.jpg)

```
//main.c
int main (int argc, char * argv[]){

    //...

#pragma omp for
        // Different threads are assigned different pixels to trace
        for (int l = 0; l < WIDTH * HEIGHT; l++) {

            //...

            // We raytrace a single pixel multiple times to account for non-deterministic scattering
            // https://en.wikipedia.org/wiki/Bidirectional_scattering_distribution_function
            for (int k = 0; k < SAMPLES_PER_PIXEL; k++) {
                CFLOAT u =
                    ((CFLOAT)i + util_randomFloat(0.0, 1.0)) / (WIDTH - 1);
                CFLOAT v =
                    ((CFLOAT)j + util_randomFloat(0.0, 1.0)) / (HEIGHT - 1);
                r = cam_getRay(&c, u, v);

                // The ray tracing function that computes the colour of the ray at the current pixel
                temp = ray_c(r, world, MAX_DEPTH);

                pcR += temp.r;
                pcG += temp.g;
                pcB += temp.b;

                alloc_linearAllocFCFreeAll(lafc);
            }

            image[i + WIDTH * (HEIGHT - 1 - j)] =
                writeColor(pcR, pcG, pcB, SAMPLES_PER_PIXEL);

            localSteps += 1;

            // synchronisation and progress tracking
            // ...


        }
```

The code snippet is the raytracing function. In theory, light rays bounces infinitively, which is infeasible to compute. 
Instead we determine a artifical limit at which we stop tracing the ray. This is the depth of recursive ray tracing 
computation.

At every trace, we need to first determine if the ray has hit any object in the world. The `obj_objLLHit` function 
loops through all the objects in the world and computes mathematically whether the current ray will eventually
intersect an objec and returns the information (location, colour, material) of the closes object it hits.

```
// main.c

RGBColorF ray_c(Ray r, const ObjectLL *world, int depth) {

    // the recursive break, stop tracing rays once we've reached the deepest recursion level
    if (depth <= 0) {
        return (RGBColorF){0};
    }

    // checks if the ray hits an object
    HitRecord rec;
    rec.valid = false;
    bool checkHit =
        obj_objLLHit(world, r, 0.00001, FLT_MAX, &rec);
```

Once a hit was registered, we then perform the lighting colour calculations at the point of intersection and generate a
new ray that gets scattered (or simply reflected in this case) in a "random" direction.

```
    // if the ray hits an object in the scene
    if (checkHit) {
        // the scattered ray
        Ray scattered = {0};
        // attenuation factor due to scattering and absorption
        RGBColorF attenuation = {0};

        // calculate the scattered ray and the attenuation factor based on the
        // material
        if (mat_scatter(&r, &rec, &attenuation, &scattered)) {

            // calcuate the colour of the scattere ray, this is the recursive function call
            RGBColorF color = ray_c(scattered, world, depth - 1);

            // multiply the colour by the attenuation factor, this crucial to mimic light 
            // losing energy as it interacts with objects in the world
            color = colorf_multiply(color, attenuation);

            return color;
        }

        return (RGBColorF){0};
    }
```

If the ray does not intersect with any object in the scene, then we simply return a background colour of our choosing.
Here we compute a blue to gray gradient that mimics that colour of the sky.

```
    // if the ray doesn't hit and object, return the background colour 
    // set the background to look like the colour of the sky
    vec3 ud = r.direction;
    vector3_normalize(&ud);
    CFLOAT t = 0.5 * (ud.y + 1.0);
    vec3 inter4;
    vector3_setf3(&inter4, 1.0 - t, 1.0 - t, 1.0 - t);
    vec3 inter3;
    vector3_setf3(&inter3, 0.5 * t, 0.7 * t, 1.0 * t);
    vector3_add(&inter3, &inter4);
    return (RGBColorF){.r = inter3.x, .g = inter3.y, .b = inter3.z};
}
```
