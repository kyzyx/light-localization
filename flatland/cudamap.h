#ifndef __CUDAMAP
#define __CUDAMAP
#include <cuda.h>
#include <vector_types.h>

typedef struct {
    // Device pointers
    float* d_intensities;
    float* d_noise;
    float4* d_surfels;
    float4* d_line_occluders;
    float4* d_circle_occluders;
    float4* d_field;
    float* d_buffer;
    float* d_tmp;
    float* d_density;
    cudaArray* d_field_tex;
    cudaArray* d_density_tex;

    // Occluder counts
    int nlines, ncircles;

    // Data bounds
    int n;
    int w,h;
    float maxx, maxy, minx, miny;
} Cudamap;

extern "C" void Cudamap_init(Cudamap* cudamap, float* surfels, float* line_occluders=NULL, float* circle_occluders=NULL);
extern "C" void Cudamap_setGLTexture(Cudamap* cudamap, unsigned int* tex);
extern "C" void Cudamap_setGLBuffer(Cudamap* cudamap, unsigned int pbo);
extern "C" void Cudamap_free(Cudamap* cudamap);
extern "C" void Cudamap_setIntensities(Cudamap* cudamap, float* intensities);
extern "C" void Cudamap_setNoise(Cudamap* cudamap, float* noise);
extern "C" void Cudamap_addLight(Cudamap* cudamap, float intensity, float x, float y);
extern "C" void Cudamap_addDirectionalLight(Cudamap* cudamap, float intensity, float x, float y, float fx, float fy);
extern "C" void Cudamap_computeField(Cudamap* cudamap, float* field);
extern "C" void Cudamap_computeDensity(Cudamap* cudamap, float* density, float threshold=5);
#endif
