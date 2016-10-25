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
__device__ static float2 cmpVI(float2 a, float2 b) {
    return a.x<b.x?a:b;
}
__device__ static unsigned long long int _float2_ll(float2 a) {
    return *((unsigned long long int*) &a);
}
__device__ static float2 _ll_float2(unsigned long long int a) {
    return *((float2*) &a);
}
__device__ static float2 atomicMin2(float2* address, float2 val)
{
    unsigned long long int* address_as_i = (unsigned long long int*) address;
    unsigned long long int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                _float2_ll(cmpVI(val, _ll_float2(assumed)))
                );
    } while (assumed != old);
    return _ll_float2(old);
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
        float2* field,
        int w, int h,
        float rangex, float rangey, float minx, float miny
        )
{
    __shared__ float2 mini[BLOCK_SIZE];

    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;
    mini[tid].x = MAX_FLOAT;
    mini[tid].y = 0;

    if (surfaceIdx < n) {
        // Data load
        float intensity = intensities[surfaceIdx];
        float4 surfel = surfels[surfaceIdx];

        int ex = 32767*((surfel.x-minx)/(rangex*w));
        int ey = 32767*((surfel.y-miny)/(rangey*h));
        mini[tid].y = __int_as_float((ex<<15)|ey);

        // Computation
        float Lx = rangex*blockIdx.y + minx - surfel.x;
        float Ly = rangey*blockIdx.z + miny - surfel.y;
        float LdotL = Lx*Lx + Ly*Ly;
        float ndotLn = (surfel.z*Lx + surfel.w*Ly)/sqrt(LdotL);
        mini[tid].x = ndotLn>0?intensity*LdotL/ndotLn:MAX_FLOAT;
    }
    __syncthreads();

    // Reduction
    if (blockSize >= 512) {
        if (tid < 256) {
            mini[tid] = mini[tid+256].x<mini[tid].x?mini[tid+256]:mini[tid];
        }
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            mini[tid] = mini[tid+128].x<mini[tid].x?mini[tid+128]:mini[tid];
        }
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            mini[tid] = mini[tid+64].x<mini[tid].x?mini[tid+64]:mini[tid];
        }
        __syncthreads(); 
    }
    if (blockSize >= 64)  {
        if (tid < 32) {
            mini[tid] = mini[tid+32].x<mini[tid].x?mini[tid+32]:mini[tid];
        }
        __syncthreads(); 
    }
    if (blockSize >= 32)  {
        if (tid < 16) {
            mini[tid] = mini[tid+16].x<mini[tid].x?mini[tid+16]:mini[tid];
        }
        __syncthreads(); 
    }
    if (blockSize >= 16)  {
        if (tid < 8) {
            mini[tid] = mini[tid+8].x<mini[tid].x?mini[tid+8]:mini[tid];
        }
        __syncthreads(); 
    }
    if (blockSize >= 8)   {
        if (tid < 4) {
            mini[tid] = mini[tid+4].x<mini[tid].x?mini[tid+4]:mini[tid];
        }
        __syncthreads(); 
    }
    if (blockSize >= 4)   {
        if (tid < 2) {
            mini[tid] = mini[tid+2].x<mini[tid].x?mini[tid+2]:mini[tid];
        }
        __syncthreads(); 
    }
    if (blockSize >= 2)   {
        if (tid < 1) {
            mini[tid] = mini[tid+1].x<mini[tid].x?mini[tid+1]:mini[tid];
        }
        __syncthreads(); 
    }

    // Final data copy
    if (tid == 0) {
        atomicMin2(field+blockIdx.z*w+blockIdx.y, mini[0]);
    }
}

void Cudamap_init(Cudamap* cudamap, float* surfels) {
    cudaSetDevice(0);
    cudaMalloc((void**) &(cudamap->d_intensities), sizeof(float)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_surfels), sizeof(float4)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_field), sizeof(float2)*cudamap->w*cudamap->h);

    cudaMemcpy(cudamap->d_surfels, surfels, sizeof(float4)*cudamap->n, cudaMemcpyHostToDevice);
    cudaMemset((void*) cudamap->d_intensities, 0, sizeof(float)*cudamap->n);
}

void Cudamap_setGLTexture(Cudamap* cudamap, unsigned int tex) {
    cudaGLSetGLDevice(0);
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
    cudaGLSetGLDevice(0);
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
    static int running = 0;
    int n = cudamap->n;
    int w = cudamap->w;
    int h = cudamap->h;

    if (running) return;
    running = 1;
    for (int i = 0; i < w*h; i++) {
        field[2*i] = MAX_FLOAT;
        field[2*i+1] = 0;
    }
    cudaMemcpy(cudamap->d_field, field, sizeof(float2)*w*h, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 blocks((n+BLOCK_SIZE-1)/BLOCK_SIZE, w, h);

    float rangex = (cudamap->maxx-cudamap->minx)/((float)w-2);
    float rangey = (cudamap->maxy-cudamap->miny)/((float)h-2);
    cuCompute<BLOCK_SIZE><<< blocks, threads >>>(
            cudamap->d_intensities,
            cudamap->d_surfels,
            n, cudamap->d_field, w, h,
            rangex, rangey, cudamap->minx, cudamap->miny
            );

    if (cudamap->d_field_tex) {
        cudaMemcpyToArray(cudamap->d_field_tex, 0, 0, cudamap->d_field, sizeof(float2)*w*h, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(field, cudamap->d_field, sizeof(float2)*w*h, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    running = 0;
}
