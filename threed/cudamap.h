#ifndef __CUDAMAP
#define __CUDAMAP
#include <cuda.h>
#include <vector_types.h>

typedef struct {
    // Device pointers
    float* d_intensities;
    float3* d_surfel_pos;
    float3* d_surfel_normal;
    float2* d_field;
    cudaArray* d_field_tex;

    // Data bounds
    int n;
    int w,h;
} Cudamap;

extern "C" void Cudamap_init(Cudamap* cudamap, float* surfels);
extern "C" void Cudamap_setGLTexture(Cudamap* cudamap, unsigned int tex);
extern "C" void Cudamap_setGLBuffer(Cudamap* cudamap, unsigned int pbo);
extern "C" void Cudamap_free(Cudamap* cudamap);
extern "C" void Cudamap_setIntensities(Cudamap* cudamap, float* intensities);
extern "C" void Cudamap_addLight(Cudamap* cudamap, float intensity, float x, float y, float z);
extern "C" void Cudamap_compute(Cudamap* cudamap, float* field, float* plane_normal, float* plane_axis, float* plane_point);
#endif
