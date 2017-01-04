#include "cudamap.h"
#include <cuda_gl_interop.h>
#include "helper_math.h"
#include <stdio.h>

#define BLOCK_SIZE 512
#define MAX_FLOAT 1e9

__device__ static float2 cmpVI(float2 a, float2 b) {
    return a.x<b.x?a:b;
}
__device__ static unsigned long long int _float2_ll(float2 a) {
    return *((unsigned long long int*) &a);
}
__device__ static float2 _ll_float2(unsigned long long int a) {
    return *((float2*) &a);
}

// From http://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
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

__global__ void cuAddlight(
        float* intensities,
        float3* surfel_pos,
        float3* surfel_normal,
        float intensity, float x, float y, float z, int n)
{
    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;

    if (surfaceIdx < n) {
        float3 pos = surfel_pos[surfaceIdx];
        float3 norm = surfel_normal[surfaceIdx];
        float3 p = make_float3(x,y,z);
        float3 L = p - pos;
        float LdotL = dot(L,L);
        float ndotL = dot(norm, L);

        float ret = LdotL>0?ndotL*intensity/(LdotL*sqrt(LdotL)):0;
        atomicAdd(intensities+surfaceIdx, ret);
    }
}

template <unsigned int blockSize>
__global__ void cuCompute(
        float* intensities,
        float3* surfel_pos,
        float3* surfel_normal,
        float3 plane_normal,
        float3 plane_axis,
        float3 plane_point,
        int n,
        float2* field,
        int w, int h
        )
{
    __shared__ float2 mini[BLOCK_SIZE];

    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;
    mini[tid] = make_float2(MAX_FLOAT, 0);

    if (surfaceIdx < n) {
        // Data load
        float intensity = intensities[surfaceIdx];
        float3 pos = surfel_pos[surfaceIdx];
        float3 norm = surfel_normal[surfaceIdx];

        mini[tid].y = __int_as_float(surfaceIdx);

        // Computation
        float3 axis2 = cross(plane_normal, plane_axis);
        float2 pix = make_float2(blockIdx.y/(float)blockDim.y, blockIdx.z/(float)blockDim.z);
        float3 p = pix.x*plane_axis + pix.y*axis2;
        float3 L = p - pos;
        float LdotL = dot(L,L);
        float ndotLn = dot(norm, L)/sqrt(LdotL);
        char occl = 1;
        float v = intensity*occl*ndotLn>0?intensity*LdotL/ndotLn:MAX_FLOAT;
        mini[tid].x = v>0.f?v:MAX_FLOAT;
    }
    __syncthreads();

    // Reduction
    if (blockSize >= 512) {
        if (tid < 256) { mini[tid] = cmpVI(mini[tid+256], mini[tid]); }
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { mini[tid] = cmpVI(mini[tid+128], mini[tid]); }
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid < 64)  { mini[tid] = cmpVI(mini[tid+64], mini[tid]); }
        __syncthreads(); 
    }
    if (blockSize >= 64)  {
        if (tid < 32)  { mini[tid] = cmpVI(mini[tid+32], mini[tid]); }
        __syncthreads(); 
    }
    if (blockSize >= 32)  {
        if (tid < 16)  { mini[tid] = cmpVI(mini[tid+16], mini[tid]); }
        __syncthreads(); 
    }
    if (blockSize >= 16)  {
        if (tid < 8)   { mini[tid] = cmpVI(mini[tid+8], mini[tid]); }
        __syncthreads(); 
    }
    if (blockSize >= 8)   {
        if (tid < 4)   { mini[tid] = cmpVI(mini[tid+4], mini[tid]); }
        __syncthreads(); 
    }
    if (blockSize >= 4)   {
        if (tid < 2)   { mini[tid] = cmpVI(mini[tid+2], mini[tid]); }
        __syncthreads(); 
    }
    if (blockSize >= 2)   {
        if (tid < 1)   { mini[tid] = cmpVI(mini[tid+1], mini[tid]); }
        __syncthreads(); 
    }

    // Final data copy
    if (tid == 0) {
        atomicMin2(field+blockIdx.z*w+blockIdx.y, mini[0]);
    }
}
void Cudamap_init(Cudamap* cudamap, const float* surfel_pos, const float* surfel_normal) {
    cudaSetDevice(0);
    cudaMalloc((void**) &(cudamap->d_intensities), sizeof(float)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_surfel_pos), sizeof(float3)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_surfel_normal), sizeof(float3)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_field), sizeof(float2)*cudamap->w*cudamap->h);

    cudaMemcpy(cudamap->d_surfel_pos, surfel_pos, sizeof(float3)*cudamap->n, cudaMemcpyHostToDevice);
    cudaMemcpy(cudamap->d_surfel_normal, surfel_normal, sizeof(float3)*cudamap->n, cudaMemcpyHostToDevice);
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
    cudaFree(cudamap->d_surfel_pos);
    cudaFree(cudamap->d_surfel_normal);
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
void Cudamap_addLight(Cudamap* cudamap, float intensity, float x, float y, float z) {
    cuAddlight<<< (cudamap->n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE >>>(
            cudamap->d_intensities,
            cudamap->d_surfel_pos,
            cudamap->d_surfel_normal,
            intensity, x, y, z, cudamap->n);
}

void Cudamap_compute(Cudamap* cudamap, float* field, const float* plane_normal, const float* plane_axis, const float* plane_point)
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

    cuCompute<BLOCK_SIZE><<< blocks, threads >>>(
            cudamap->d_intensities,
            cudamap->d_surfel_pos,
            cudamap->d_surfel_normal,
            make_float3(plane_normal[0], plane_normal[1], plane_normal[2]),
            make_float3(plane_axis[0], plane_axis[1], plane_axis[2]),
            make_float3(plane_point[0], plane_point[1], plane_point[2]),
            n, cudamap->d_field, w, h
            );

    if (cudamap->d_field_tex) {
        cudaMemcpyToArray(cudamap->d_field_tex, 0, 0, cudamap->d_field, sizeof(float2)*w*h, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(field, cudamap->d_field, sizeof(float2)*w*h, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    running = 0;
}
