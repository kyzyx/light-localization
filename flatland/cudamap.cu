#include "cudamap.h"
#include <cuda_gl_interop.h>
#include <stdio.h>

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

__global__ void cuAddlight(float* intensities, float4* surfels, float intensity, float x, float y, int n)
{
    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;
    if (surfaceIdx < n) {
        float4 surfel = surfels[surfaceIdx];

        float2 L;
        L.x = x - surfel.x;
        L.y = y - surfel.y;

        float LdotL = L.x*L.x+L.y*L.y;
        float ndotL = fmaxf(surfel.z*L.x+surfel.w*L.y,0.f);
        float ret = LdotL>0?ndotL*intensity/(LdotL*sqrt(LdotL)):0;
        atomicAdd(intensities+surfaceIdx, ret);
    }
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
    cudaSetDevice(0);
    cudaGLSetGLDevice(0);
    cudaMalloc((void**) &(cudamap->d_intensities), sizeof(float)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_surfels), sizeof(float4)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_field), sizeof(float)*cudamap->w*cudamap->h);

    cudaMemcpy(cudamap->d_surfels, surfels, sizeof(float4)*cudamap->n, cudaMemcpyHostToDevice);
    cudaMemset((void*) cudamap->d_intensities, 0, sizeof(float)*cudamap->n);
}

void Cudamap_setGLTexture(Cudamap* cudamap, unsigned int tex) {
    cudaStream_t cuda_stream;
    cudaGraphicsResource *resources[1];

    cudaGraphicsGLRegisterImage(resources, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaStreamCreate(&cuda_stream);
    cudaGraphicsMapResources(1, resources, cuda_stream);
    cudaGraphicsSubResourceGetMappedArray(&(cudamap->d_field_tex), resources[0], 0, 0);
    cudaGraphicsUnmapResources(1, resources, cuda_stream);
    cudaStreamDestroy(cuda_stream);
}

void Cudamap_setGLBuffer(Cudamap* cudamap, unsigned int pbo) {
    cudaStream_t cuda_stream;
    cudaGraphicsResource *resources[1];
    size_t size;

    cudaGraphicsGLRegisterBuffer(resources, pbo, cudaGraphicsMapFlagsNone);
    cudaStreamCreate(&cuda_stream);
    cudaGraphicsMapResources(1, resources, cuda_stream);
    cudaGraphicsResourceGetMappedPointer((void **)&(cudamap->d_field), &size, resources[0]);
    cudaGraphicsUnmapResources(1, resources, cuda_stream);
    cudaStreamDestroy(cuda_stream);
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

void Cudamap_addLight(Cudamap* cudamap, float intensity, float x, float y) {
    cuAddlight<<< (cudamap->n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE >>>(
            cudamap->d_intensities, cudamap->d_surfels, intensity, x, y, cudamap->n);
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

    if (cudamap->d_field_tex) {
        cudaMemcpyToArray(cudamap->d_field_tex, 0, 0, cudamap->d_field, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(field, cudamap->d_field, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
}
