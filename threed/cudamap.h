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

    // Data bounds
    int n;
    int w;
} Cudamap;

extern "C" void Cudamap_init(Cudamap* cudamap, const float* surfel_pos, const float* surfel_normal);
extern "C" void Cudamap_free(Cudamap* cudamap);
extern "C" void Cudamap_setIntensities(Cudamap* cudamap, float* intensities);
extern "C" void Cudamap_addLight(Cudamap* cudamap, float intensity, float x, float y, float z);
extern "C" void Cudamap_compute(Cudamap* cudamap, float* field);
#endif
