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
        float* intensities,
        float4* surfels,
        int n,
        float* field,
        int w, int h,
        float maxx, float maxy, float minx, float miny
        )
{
    __shared__ float mini[BLOCK_SIZE];

    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;
    float x = (maxx - minx)*(blockIdx.y+1.5)/((float)w-2) + minx;
    float y = (maxy - miny)*(blockIdx.z+1.5)/((float)h-2) + miny;
    int pointIdx = blockIdx.z*w+blockIdx.y;
    mini[tid] = MAX_FLOAT;

    if (surfaceIdx < n) {
        // Data load
        float intensity = intensities[surfaceIdx];
        float4 surfel = surfels[surfaceIdx];

        // Computation
        float Lx = x - surfel.x;
        float Ly = y - surfel.y;
        float ndotL = surfel.z*Lx + surfel.w*Ly;
        float LdotL = Lx*Lx + Ly*Ly;
        float mag = sqrt(LdotL);
        Lx /= mag;
        Ly /= mag;
        float ndotLn = surfel.z*Lx + surfel.w*Ly;
        mini[tid] = ndotL>1e-9?intensity*LdotL/ndotLn:MAX_FLOAT;
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
        float* intensities,
        float* surfels,
        int n,
        float* field,
        int w, int h,
        float maxx, float maxy, float minx, float miny
        )
{
    float* d_intensities;
    float4* d_surfels;
    float* d_field;

    cudaMalloc((void**) &d_intensities, sizeof(float)*n);
    cudaMalloc((void**) &d_surfels, sizeof(float4)*n);
    cudaMalloc((void**) &d_field, sizeof(float)*w*h);

    cudaMemcpy(d_intensities, intensities, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_surfels, surfels, sizeof(float4)*n, cudaMemcpyHostToDevice);
    for (int i = 0; i < w*h; i++) field[i] = MAX_FLOAT;
    cudaMemcpy(d_field, field, sizeof(float)*w*h, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 blocks((n+BLOCK_SIZE-1)/BLOCK_SIZE, w, h);

    cuCompute<BLOCK_SIZE><<< blocks, threads >>>(
            d_intensities,
            d_surfels,
            n,
            d_field,
            w, h,
            maxx, maxy, minx, miny
            );

    cudaMemcpy(field, d_field, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
    cudaFree(d_intensities);
    cudaFree(d_surfels);
    cudaFree(d_field);
}
