#include "cudamap.h"
#include "convolutionSeparable.h"
#include <cuda_gl_interop.h>
#include <stdio.h>

#define BLOCK_SIZE 512
#define MAX_FLOAT 1e9

__host__ __device__ int cdist(int a, int b, int maxidx) {
    int d = abs(b-a);
    d = d>maxidx/2?maxidx-d:d;
    return b>a?d:-d;
}

inline __host__ __device__ float2 normalize(float2 a) {
    float l = sqrt(a.x*a.x + a.y*a.y);
    return make_float2(a.x/l, a.y/l);
}
inline __host__ __device__ float dot(float2 a, float2 b) {
        return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
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

__device__ static float ccw(float2 a, float2 b, float2 c) {
    return (b.x - a.x)*(c.y - a.y) - (c.x - a.x)*(b.y - a.y);
}
__device__ static char intersects(float2 a, float2 b, float2 c, float2 d) {
    char v1 = ccw(a, b, c)*ccw(a, b, d) > 0?0:1;
    char v2 = ccw(c, d, a)*ccw(c, d, b) > 0?0:1;
    return v1*v2;
}
__device__ static char lineocclusion(float2* line_occluders, int nlines, float2 a, float2 b) {
    char occ = 1;
    for (int i = 0; i < nlines; i+=2) {
        occ *= (1-intersects(line_occluders[i], line_occluders[i+1], a, b));
    }
    return occ;
}
__device__ static char circleocclusion(float4* circle_occluders, int ncircles, float2 a, float2 b) {
    char occ = 1;
    float2 v = make_float2(b.x-a.x, b.y-a.y);
    float L = sqrt(v.x*v.x + v.y*v.y);
    v.x /= L;
    v.y /= L;
    for (int i = 0; i < ncircles; i++) {
        float2 O = make_float2(circle_occluders[i].x, circle_occluders[i].y);
        float rr = circle_occluders[i].z*circle_occluders[i].z;
        float2 EO = make_float2(O.x-a.x, O.y-a.y);
        float R = dot(EO,EO);
        float vv = dot(EO, v);
        float disc = rr - (R - vv*vv);
        float d = sqrt(disc);
        /*float t = R>rr?vv+d:vv-d;*/
        /*float2 p = make_float2(a.x+t*v.x, a.y+t*v.y);*/
        occ *= ((R>rr && (vv<=0 || L<vv-d)) || disc < 0 || (R<rr && L<vv+d))?1:0;
    }
    return occ;
}

const float EPSILON = 1e-5;

__global__ void cuAddlight(
        float* intensities,
        float4* surfels,
        float4* line_occluders, int nlines,
        float4* circle_occluders, int ncircles,
        float intensity, float x, float y, int n)
{
    __shared__ float2 shared_line_occluders[64];
    __shared__ float4 shared_circle_occluders[32];

    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;

    if (tid < nlines) {
        float4 line = line_occluders[tid];
        shared_line_occluders[2*tid] = make_float2(line.x, line.y);
        shared_line_occluders[2*tid+1] = make_float2(line.z, line.w);
    }
    if (tid < ncircles) {
        shared_circle_occluders[tid] = circle_occluders[tid];
    }
    __syncthreads();

    if (surfaceIdx < n) {
        float4 surfel = surfels[surfaceIdx];
        float2 L = make_float2(x - surfel.x, y - surfel.y);

        float LdotL = L.x*L.x+L.y*L.y;
        float ndotL = fmaxf(surfel.z*L.x+surfel.w*L.y,0.f);
        float ret = LdotL>0?ndotL*intensity/(LdotL*sqrt(LdotL)):0;
        float2 A = make_float2(surfel.x + EPSILON*surfel.z, surfel.y + EPSILON*surfel.w);
        float2 B = make_float2(x,y);
        char occl = lineocclusion(shared_line_occluders, nlines*2,A,B)*
                    circleocclusion(shared_circle_occluders,ncircles,A,B);
        atomicAdd(intensities+surfaceIdx, ret*occl);
    }
}
__global__ void cuAddDirectionalLight(
        float* intensities,
        float4* surfels,
        float4* line_occluders, int nlines,
        float4* circle_occluders, int ncircles,
        float intensity, float x, float y,
        float nx, float ny, float d, int n)
{
    __shared__ float2 shared_line_occluders[64];
    __shared__ float4 shared_circle_occluders[32];

    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;

    if (tid < nlines) {
        float4 line = line_occluders[tid];
        shared_line_occluders[2*tid] = make_float2(line.x, line.y);
        shared_line_occluders[2*tid+1] = make_float2(line.z, line.w);
    }
    if (tid < ncircles) {
        shared_circle_occluders[tid] = circle_occluders[tid];
    }
    __syncthreads();

    if (surfaceIdx < n) {
        float4 surfel = surfels[surfaceIdx];
        float2 L = make_float2(x - surfel.x, y - surfel.y);

        float LdotL = L.x*L.x+L.y*L.y;
        float ndotL = fmaxf(surfel.z*L.x+surfel.w*L.y,0.f);

        float mag = sqrt(LdotL);
        float ct = -(L.x*nx + L.y*ny)/mag;
        float scaling = ct>0.2?pow(ct, d):0;
        float ret = LdotL>0?ndotL*intensity*scaling/(LdotL*mag):0;
        float2 A = make_float2(surfel.x + EPSILON*surfel.z, surfel.y + EPSILON*surfel.w);
        float2 B = make_float2(x,y);
        char occl = lineocclusion(shared_line_occluders, nlines*2,A,B)*
                    circleocclusion(shared_circle_occluders,ncircles,A,B);
        atomicAdd(intensities+surfaceIdx, ret*occl);
    }
}

template <unsigned int blockSize>
__global__ void cuGetRanges(
        float* intensities,
        float* noise,
        float4* surfels,
        float4* line_occluders,
        int nlines,
        float4* circle_occluders,
        int ncircles,
        int n,
        float4* field,
        float noisemag,
        int w, int h,
        float rangex, float rangey, float minx, float miny
        )
{
    __shared__ float4 val;
    __shared__ int2 ranges[BLOCK_SIZE];
    __shared__ float2 shared_line_occluders[64];
    __shared__ float4 shared_circle_occluders[32];

    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;
    ranges[tid] = make_int2(0, 0);

    if (tid == 0) {
        val = field[blockIdx.z*w+blockIdx.y];
    }
    if (tid < nlines) {
        float4 line = line_occluders[tid];
        shared_line_occluders[2*tid] = make_float2(line.x, line.y);
        shared_line_occluders[2*tid+1] = make_float2(line.z, line.w);
    }
    if (tid < ncircles) {
        shared_circle_occluders[tid] = circle_occluders[tid];
    }
    __syncthreads();

    if (surfaceIdx < n) {
        // Data load
        float intensity = intensities[surfaceIdx];
        float d = noise[surfaceIdx];
        float4 surfel = surfels[surfaceIdx];

        // Computation
        float2 p = make_float2(rangex*blockIdx.y + minx, rangey*blockIdx.z + miny);
        float Lx = p.x - surfel.x;
        float Ly = p.y - surfel.y;
        float LdotL = Lx*Lx + Ly*Ly;
        float ndotLn = (surfel.z*Lx + surfel.w*Ly)/sqrt(LdotL);
        float2 A = make_float2(surfel.x + EPSILON*surfel.z, surfel.y + EPSILON*surfel.w);
        char occl = lineocclusion(shared_line_occluders, nlines*2,A,p)*
                    circleocclusion(shared_circle_occluders,ncircles,A,p);
        float v = intensity*occl*ndotLn>0?d*intensity*LdotL/ndotLn:MAX_FLOAT;
        float q = val.x == MAX_FLOAT?0:2*noisemag*val.x/(1-noisemag) - (v-val.x);
        int delta = cdist(__float_as_int(val.y), surfaceIdx, n);
        ranges[tid].x = q>0?delta:0;
        ranges[tid].y = q>0?delta:0;
    }
    __syncthreads();

    // Reduction
    if (blockSize >= 512) {
        if (tid < 256) {
            ranges[tid].x = min(ranges[tid].x, ranges[tid+256].x);
            ranges[tid].y = max(ranges[tid].y, ranges[tid+256].y);
        }
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            ranges[tid].x = min(ranges[tid].x, ranges[tid+128].x);
            ranges[tid].y = max(ranges[tid].y, ranges[tid+128].y);
        }
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid < 64)  {
            ranges[tid].x = min(ranges[tid].x, ranges[tid+64].x);
            ranges[tid].y = max(ranges[tid].y, ranges[tid+64].y);
        }
        __syncthreads(); 
    }
    if (blockSize >= 64)  {
        if (tid < 32)  {
            ranges[tid].x = min(ranges[tid].x, ranges[tid+32].x);
            ranges[tid].y = max(ranges[tid].y, ranges[tid+32].y);
        }
        __syncthreads(); 
    }
    if (blockSize >= 32)  {
        if (tid < 16)  {
            ranges[tid].x = min(ranges[tid].x, ranges[tid+16].x);
            ranges[tid].y = max(ranges[tid].y, ranges[tid+16].y);
        }
        __syncthreads(); 
    }
    if (blockSize >= 16)  {
        if (tid < 8)   {
            ranges[tid].x = min(ranges[tid].x, ranges[tid+8].x);
            ranges[tid].y = max(ranges[tid].y, ranges[tid+8].y);
        }
        __syncthreads(); 
    }
    if (blockSize >= 8)   {
        if (tid < 4)   {
            ranges[tid].x = min(ranges[tid].x, ranges[tid+4].x);
            ranges[tid].y = max(ranges[tid].y, ranges[tid+4].y);
        }
        __syncthreads(); 
    }
    if (blockSize >= 4)   {
        if (tid < 2)   {
            ranges[tid].x = min(ranges[tid].x, ranges[tid+2].x);
            ranges[tid].y = max(ranges[tid].y, ranges[tid+2].y);
        }
        __syncthreads(); 
    }
    if (blockSize >= 2)   {
        if (tid < 1)   {
            ranges[tid].x = min(ranges[tid].x, ranges[tid+1].x);
            ranges[tid].y = max(ranges[tid].y, ranges[tid+1].y);
        }
        __syncthreads(); 
    }

    // Final data copy
    if (tid == 0) {
        atomicMin((int*) &(field[blockIdx.z*w+blockIdx.y].z), ranges[0].x);
        atomicMax((int*) &(field[blockIdx.z*w+blockIdx.y].w), ranges[0].y);
    }
}

template <unsigned int blockSize>
__global__ void cuCompute(
        float* intensities,
        float* noise,
        float4* surfels,
        float4* line_occluders,
        int nlines,
        float4* circle_occluders,
        int ncircles,
        int n,
        float4* field,
        int w, int h,
        float rangex, float rangey, float minx, float miny
        )
{
    __shared__ float2 mini[BLOCK_SIZE];
    __shared__ float2 shared_line_occluders[64];
    __shared__ float4 shared_circle_occluders[32];

    int tid = threadIdx.x;
    int surfaceIdx = tid + blockDim.x*blockIdx.x;
    mini[tid] = make_float2(MAX_FLOAT, 0);

    if (tid < nlines) {
        float4 line = line_occluders[tid];
        shared_line_occluders[2*tid] = make_float2(line.x, line.y);
        shared_line_occluders[2*tid+1] = make_float2(line.z, line.w);
    }
    if (tid < ncircles) {
        shared_circle_occluders[tid] = circle_occluders[tid];
    }
    __syncthreads();

    if (surfaceIdx < n) {
        // Data load
        float intensity = intensities[surfaceIdx];
        float d = noise[surfaceIdx];
        float4 surfel = surfels[surfaceIdx];

        /*int ex = 32767*((surfel.x-minx)/(rangex*w));*/
        /*int ey = 32767*((surfel.y-miny)/(rangey*h));*/
        /*mini[tid].y = __int_as_float((ex<<15)|ey);*/
        mini[tid].y = __int_as_float(surfaceIdx);

        // Computation
        float2 p = make_float2(rangex*blockIdx.y + minx, rangey*blockIdx.z + miny);
        float Lx = p.x - surfel.x;
        float Ly = p.y - surfel.y;
        float LdotL = Lx*Lx + Ly*Ly;
        float ndotLn = (surfel.z*Lx + surfel.w*Ly)/sqrt(LdotL);
        float2 A = make_float2(surfel.x + EPSILON*surfel.z, surfel.y + EPSILON*surfel.w);
        char occl = lineocclusion(shared_line_occluders, nlines*2,A,p)*
                    circleocclusion(shared_circle_occluders,ncircles,A,p);
        float v = intensity*occl*ndotLn>0?d*intensity*LdotL/ndotLn:MAX_FLOAT;
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
        atomicMin2((float2*) &(field[blockIdx.z*w+blockIdx.y]), mini[0]);
    }
}

#include "adj.gen.h"
#define LOADDATA(a,x,y,w,h) (((x)<(w))&&((y)<(h)))?a[(y)*w+(x)]:make_float4(0,0,0,0)

__global__ void cuComputeDensity(
        float4* field,
        float* density,
        int n, int w, int h,
        float threshold
)
{
    __shared__ float4 data[32][32];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    // Load data
    data[ty][tx]       = LOADDATA(field,x,y,w,h);
    data[ty+16][tx]    = LOADDATA(field,x,y+16,w,h);
    data[ty][tx+16]    = LOADDATA(field,x+16,y,w,h);
    data[ty+16][tx+16] = LOADDATA(field,x+16,y+16,w,h);
    __syncthreads();

    x += 8;
    y += 8;
    if (x < w-1 && y < h-1) {
        tx += 8;
        ty += 8;
        float ret = 0;
        int count = 0;
        float4 prev = data[ty+adjy[0]][tx+adjx[0]];
        for (int i = 1; i < NUM_ADJ; i++) {
            float4 curr = data[ty+adjy[i]][tx+adjx[i]];
            int a = __float_as_int(prev.y);
            int ahi = __float_as_int(prev.w);
            int alo = __float_as_int(prev.z);
            int b = __float_as_int(curr.y);
            int bhi = __float_as_int(curr.w);
            int blo = __float_as_int(curr.z);
            int a1 = max(a+ahi, b+bhi);
            int b1 = min(a+alo, b+blo);
            int d = abs(a1-b1);
            d = d>n/2?n-d:d;
            if (d < threshold) {
                ret += d;
                count += 1;
            }
            prev = curr;
        }
        density[w*y+x] = count>0?0.5*ret/count:0;
    }
}

float Gaussian(float x, float std) {
    return exp(-(x*x)/(2*std*std))/(sqrt(2*M_PI)*std);
}

void Cudamap_init(Cudamap* cudamap, float* surfels, float* line_occluders, float* circle_occluders) {
    cudaSetDevice(0);
    cudaMalloc((void**) &(cudamap->d_intensities), sizeof(float)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_noise), sizeof(float)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_surfels), sizeof(float4)*cudamap->n);
    cudaMalloc((void**) &(cudamap->d_field), sizeof(float4)*cudamap->w*cudamap->h);
    cudaMalloc((void**) &(cudamap->d_density), sizeof(float)*cudamap->w*cudamap->h);
    cudaMalloc((void**) &(cudamap->d_buffer), sizeof(float)*cudamap->w*cudamap->h);
    cudaMalloc((void**) &(cudamap->d_tmp), sizeof(float)*cudamap->w*cudamap->h);
    cudaMalloc((void**) &(cudamap->d_line_occluders), sizeof(float4)*cudamap->nlines);
    cudaMalloc((void**) &(cudamap->d_circle_occluders), sizeof(float4)*cudamap->ncircles);

    cudaMemcpy(cudamap->d_surfels, surfels, sizeof(float4)*cudamap->n, cudaMemcpyHostToDevice);
    if (cudamap->nlines) cudaMemcpy(cudamap->d_line_occluders, line_occluders, sizeof(float4)*cudamap->nlines, cudaMemcpyHostToDevice);
    if (cudamap->ncircles) cudaMemcpy(cudamap->d_circle_occluders, circle_occluders, sizeof(float4)*cudamap->ncircles, cudaMemcpyHostToDevice);
    cudaMemset((void*) cudamap->d_intensities, 0, sizeof(float)*cudamap->n);
    cudaMemset((void*) cudamap->d_noise, 0, sizeof(float)*cudamap->n);

    float kernel[KERNEL_LENGTH];
    float tot = 0;
    for (int i = 0; i <= KERNEL_RADIUS; i++) {
        kernel[KERNEL_RADIUS-i] = Gaussian(i, 2.5);
        kernel[KERNEL_RADIUS+i] = kernel[KERNEL_RADIUS-i];
    }
    for (int i = 0; i < KERNEL_LENGTH; i++) {
        tot += kernel[i];
    }
    for (int i = 0; i < KERNEL_LENGTH; i++) {
        kernel[i] /= tot;
    }

    setConvolutionKernel(kernel);
}

void Cudamap_setGLTexture(Cudamap* cudamap, unsigned int* tex) {
    cudaGLSetGLDevice(0);
    cudaStream_t cuda_stream;
    cudaGraphicsResource *resources[1];

    cudaGraphicsGLRegisterImage(resources, tex[0], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaStreamCreate(&cuda_stream);
    cudaGraphicsMapResources(1, resources, cuda_stream);
    cudaGraphicsSubResourceGetMappedArray(&(cudamap->d_field_tex), resources[0], 0, 0);
    cudaGraphicsUnmapResources(1, resources, cuda_stream);
    cudaStreamDestroy(cuda_stream);

    cudaGraphicsGLRegisterImage(resources, tex[1], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaStreamCreate(&cuda_stream);
    cudaGraphicsMapResources(1, resources, cuda_stream);
    cudaGraphicsSubResourceGetMappedArray(&(cudamap->d_density_tex), resources[0], 0, 0);
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
    cudaFree(cudamap->d_buffer);
    cudaFree(cudamap->d_tmp);
    cudaFree(cudamap->d_intensities);
    cudaFree(cudamap->d_noise);
    cudaFree(cudamap->d_field);
}

void Cudamap_setIntensities(Cudamap* cudamap, float* intensities) {
    if (intensities) {
        cudaMemcpy(cudamap->d_intensities, intensities, sizeof(float)*cudamap->n, cudaMemcpyHostToDevice);
    } else {
        cudaMemset((void*) cudamap->d_intensities, 0, sizeof(float)*cudamap->n);
    }
}
void Cudamap_setNoise(Cudamap* cudamap, float* noise) {
    if (noise) {
        cudaMemcpy(cudamap->d_noise, noise, sizeof(float)*cudamap->n, cudaMemcpyHostToDevice);
    } else {
        cudaMemset((void*) cudamap->d_noise, 0, sizeof(float)*cudamap->n);
    }
}

void Cudamap_addLight(Cudamap* cudamap, float intensity, float x, float y) {
    cuAddlight<<< (cudamap->n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE >>>(
            cudamap->d_intensities, cudamap->d_surfels,
            cudamap->d_line_occluders, cudamap->nlines,
            cudamap->d_circle_occluders, cudamap->ncircles,
            intensity, x, y, cudamap->n);
}
void Cudamap_addDirectionalLight(Cudamap* cudamap, float intensity, float x, float y, float fx, float fy) {
    float d = sqrt(fx*fx + fy*fy);
    cuAddDirectionalLight<<< (cudamap->n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE >>>(
            cudamap->d_intensities, cudamap->d_surfels,
            cudamap->d_line_occluders, cudamap->nlines,
            cudamap->d_circle_occluders, cudamap->ncircles,
            intensity, x, y, fx/d, fy/d, d, cudamap->n);
}

void Cudamap_computeField(Cudamap* cudamap, float* field, float noisetolerance)
{
    static int running = 0;
    int n = cudamap->n;
    int w = cudamap->w;
    int h = cudamap->h;

    if (running) return;
    running = 1;
    for (int i = 0; i < w*h; i++) {
        field[4*i] = MAX_FLOAT;
        field[4*i+1] = 0;
        field[4*i+2] = 0;
        field[4*i+3] = 0;
    }
    cudaMemcpy(cudamap->d_field, field, sizeof(float4)*w*h, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 blocks((n+BLOCK_SIZE-1)/BLOCK_SIZE, w, h);

    float rangex = (cudamap->maxx-cudamap->minx)/((float)w-2);
    float rangey = (cudamap->maxy-cudamap->miny)/((float)h-2);
    cuCompute<BLOCK_SIZE><<< blocks, threads >>>(
            cudamap->d_intensities,
            cudamap->d_noise,
            cudamap->d_surfels,
            cudamap->d_line_occluders,
            cudamap->nlines,
            cudamap->d_circle_occluders,
            cudamap->ncircles,
            n, cudamap->d_field, w, h,
            rangex, rangey, cudamap->minx, cudamap->miny
            );

    cuGetRanges<BLOCK_SIZE><<< blocks, threads >>>(
            cudamap->d_intensities,
            cudamap->d_noise,
            cudamap->d_surfels,
            cudamap->d_line_occluders,
            cudamap->nlines,
            cudamap->d_circle_occluders,
            cudamap->ncircles,
            n, cudamap->d_field, noisetolerance, w, h,
            rangex, rangey, cudamap->minx, cudamap->miny
            );

    if (cudamap->d_field_tex) {
        cudaMemcpyToArray(cudamap->d_field_tex, 0, 0, cudamap->d_field, sizeof(float4)*w*h, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(field, cudamap->d_field, sizeof(float4)*w*h, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    running = 0;
}

void Cudamap_computeDensity(Cudamap* cudamap, float* density, float threshold)
{
    int n = cudamap->n;
    int w = cudamap->w;
    int h = cudamap->h;
    cudaMemset(cudamap->d_buffer, 0, sizeof(float)*w*h);
    cudaMemset(cudamap->d_tmp, 0, sizeof(float)*w*h);
    cudaMemset(cudamap->d_density, 0, sizeof(float)*w*h);

    dim3 threads(16, 16, 1);
    dim3 blocks((w+15)/16-1, (h+15)/16-1, 1);

    cuComputeDensity<<< blocks, threads >>>(
        cudamap->d_field, cudamap->d_buffer, n, w, h, threshold
        /*cudamap->d_field, cudamap->d_density, n, w, h, threshold*/
    );
    convolutionRowsGPU(cudamap->d_tmp, cudamap->d_buffer, w, h);
    convolutionColumnsGPU(cudamap->d_density, cudamap->d_tmp, w, h);
    if (cudamap->d_density_tex) {
        cudaMemcpyToArray(cudamap->d_density_tex, 0, 0, cudamap->d_density, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(density, cudamap->d_density, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
