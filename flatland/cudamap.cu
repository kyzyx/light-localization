#include "cudamap.h"
#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 512
#define MAX_FLOAT 1e4

// From http://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template <unsigned int blockSize>
__global__ void cuCompute(
        surfel* surfels,
        int n,
        float* field,
        int w, int h,
        float maxx, float maxy, float minx, float miny
        )
{
    surfel s;
    __shared__ float mini[BLOCK_SIZE];

    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;
    float x = (maxx - minx)*(blockIdx.y+1.5)/((float)w-2) + minx;
    float y = (maxy - miny)*(blockIdx.z+1.5)/((float)h-2) + miny;
    int pointIdx = blockIdx.z*w+blockIdx.y;
    mini[tid] = MAX_FLOAT;

    // Data load
    s.intensity = surfels[surfaceIdx].intensity;
    s.coords[0] = surfels[surfaceIdx].coords[0];
    s.coords[1] = surfels[surfaceIdx].coords[1];
    s.normal[0] = surfels[surfaceIdx].normal[0];
    s.normal[1] = surfels[surfaceIdx].normal[1];
    // Computation
    if (surfaceIdx < n) {
        float Lx = x - s.coords[0];
        float Ly = y - s.coords[1];
        float ndotL = s.normal[0]*Lx + s.normal[1]*Ly;
        float LdotL = Lx*Lx + Ly*Ly;
        float mag = sqrt(LdotL);
        Lx /= mag;
        Ly /= mag;
        float ndotLn = s.normal[0]*Lx + s.normal[1]*Ly;
        mini[tid] = ndotL>1e-9?s.intensity*LdotL/ndotLn:MAX_FLOAT;
    }
    __syncthreads();

    // Reduction
    if (blockSize >= 512) { if (tid < 256) mini[tid] = fminf(mini[tid + 256], mini[tid]); __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) mini[tid] = fminf(mini[tid + 128], mini[tid]); __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) mini[tid] = fminf(mini[tid + 64], mini[tid]);   __syncthreads(); }
    if (blockSize >= 64)  { if (tid < 32) mini[tid] = fminf(mini[tid + 32], mini[tid]);   __syncthreads(); }
    if (blockSize >= 32)  { if (tid < 16) mini[tid] = fminf(mini[tid + 16], mini[tid]);   __syncthreads(); }
    if (blockSize >= 16)  { if (tid < 8) mini[tid] = fminf(mini[tid + 8], mini[tid]);     __syncthreads(); }
    if (blockSize >= 8)   { if (tid < 4) mini[tid] = fminf(mini[tid + 4], mini[tid]);     __syncthreads(); }
    if (blockSize >= 4)   { if (tid < 2) mini[tid] = fminf(mini[tid + 2], mini[tid]);     __syncthreads(); }
    if (blockSize >= 2)   { if (tid < 1) mini[tid] = fminf(mini[tid + 1], mini[tid]);     __syncthreads(); }

    // Final data copy
    if (tid == 0) {
        atomicMin(field+pointIdx, mini[tid]);
    }
}

void computemap_cuda(
        surfel* surfels,
        int n,
        float* field,
        int w, int h,
        float maxx, float maxy, float minx, float miny
        )
{
    surfel* d_surfels;
    float* d_field;

    cudaMalloc((void**) &d_surfels, sizeof(surfel)*n);
    cudaMalloc((void**) &d_field, sizeof(float)*w*h);

    cudaMemcpy(d_surfels, surfels, sizeof(surfel)*n, cudaMemcpyHostToDevice);
    for (int i = 0; i < w*h; i++) field[i] = MAX_FLOAT;
    cudaMemcpy(d_field, field, sizeof(float)*w*h, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 blocks((n+BLOCK_SIZE-1)/BLOCK_SIZE, w, h);

    cuCompute<BLOCK_SIZE><<< blocks, threads >>>(
            d_surfels,
            n,
            d_field,
            w, h,
            maxx, maxy, minx, miny
            );

    cudaMemcpy(field, d_field, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
    cudaFree(d_surfels);
    cudaFree(d_field);
}
