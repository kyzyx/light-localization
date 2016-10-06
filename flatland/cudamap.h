#ifndef __CUDAMAP
#define __CUDAMAP

typedef struct {
    float intensity;
    short pixcoords[2];
    float coords[2];
    float normal[2];
} surfel;

extern "C" void computemap_cuda(
        surfel* surfels,
        int n,
        float* field,
        int w, int h,
        float maxx, float maxy, float minx, float miny
        );
#endif
