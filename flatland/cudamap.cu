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

void Cudamap_init(Cudamap* cudamap, float* surfels) {
    cudaMalloc((void**) &(cudamap->d_intensities), sizeof(float)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_surfels), sizeof(float4)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_field), sizeof(float)*cudamap->w*cudamap->h);

    cudaMemcpy(cudamap->d_surfels, surfels, sizeof(float4)*cudamap->n, cudaMemcpyHostToDevice);
    cudaMemset((void*) cudamap->d_intensities, 0, sizeof(float)*cudamap->n);
}

void Cudamap_free(Cudamap* cudamap) {
    cudaFree(cudamap->d_surfels);
    cudaFree(cudamap->d_intensities);
    cudaFree(cudamap->d_field);
}

void Cudamap_setIntensities(Cudamap* cudamap, float* intensities) {
    if (intensities) {
        cudaMemcpy(cudamap->d_intensities, intensities, sizeof(float)*cudamap->n, cudaMemcpyHostToDevice);
    } else {
        cudaMemset((void*) cudamap->d_intensities, 0, sizeof(float)*cudamap->n);
    }
}

void Cudamap_compute(Cudamap* cudamap, float* field)
{
    int n = cudamap->n;
    int w = cudamap->w;
    int h = cudamap->h;

    for (int i = 0; i < w*h; i++) field[i] = MAX_FLOAT;
    cudaMemcpy(cudamap->d_field, field, sizeof(float)*w*h, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 blocks((n+BLOCK_SIZE-1)/BLOCK_SIZE, w, h);

    cuCompute<BLOCK_SIZE><<< blocks, threads >>>(
            cudamap->d_intensities,
            cudamap->d_surfels,
            n,
            cudamap->d_field,
            w, h,
            cudamap->maxx, cudamap->maxy, cudamap->minx, cudamap->miny
            );

    cudaMemcpy(field, cudamap->d_field, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
}
