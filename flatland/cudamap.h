#ifndef __CUDAMAP
#define __CUDAMAP
#include <cuda.h>
#include <vector_types.h>

typedef struct {
    // Device pointers
    float* d_intensities;
    float4* d_surfels;
    float* d_field;
    cudaGraphicsResource* tex;

    // Data bounds
    int n;
    int w,h;
    float maxx, maxy, minx, miny;
} Cudamap;

extern "C" void Cudamap_init(Cudamap* cudamap, float* surfels);
extern "C" void Cudamap_setGLTexture(Cudamap* cudamap, unsigned int pbo);
extern "C" void Cudamap_free(Cudamap* cudamap);
extern "C" void Cudamap_setIntensities(Cudamap* cudamap, float* intensities);
extern "C" void Cudamap_addLight(Cudamap* cudamap, float intensity, float x, float y);
extern "C" void Cudamap_compute(Cudamap* cudamap, float* field);
#endif
