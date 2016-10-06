#ifndef __CUDAMAP
#define __CUDAMAP

extern "C" void computemap_cuda(
        float* intensities,
        float* surfels,
        int n,
        float* field,
        int w, int h,
        float maxx, float maxy, float minx, float miny
        );
#endif
