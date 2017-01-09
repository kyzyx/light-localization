#include "extract.h"
#include <glm/glm.hpp>
#include <cmath>
#include <mutex>
#include <cstring>

using namespace glm;

float angle(vec3 a, vec3 b) {
    return atan(length(cross(a,b))/dot(a,b));
}

vec3 getNorm(float* df, int dim, ivec3 c) {
    float ff = df[2*(c.x*dim*dim+c.y*dim+c.z)+1];
    int ispherical = floatBitsToInt(ff);
    int itheta = ispherical&((1<<15)-1);
    int iphi = ispherical>>15;
    float theta = itheta*M_PI/(1<<15);
    float phi = iphi*2*M_PI/(1<<15);
    return vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
}

float medialaxis(float* df, int dim, ivec3 stu, float threshold) {
    float ret = 0;
    vec3 v = getNorm(df, dim, stu);
    ret += angle(v, getNorm(df, dim, stu+ivec3(-1,-1,-1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(-1,-1,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(-1,-1,1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(-1,0,-1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(-1,0,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(-1,0,1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(-1,1,-1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(-1,1,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(-1,1,1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(0,-1,-1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(0,-1,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(0,-1,1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(0,0,-1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(0,0,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(0,0,1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(0,1,-1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(0,1,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(0,1,1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(1,-1,-1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(1,-1,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(1,-1,1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(1,0,-1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(1,0,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(1,0,1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(1,1,-1)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(1,1,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(df, dim, stu+ivec3(1,1,1)))>threshold?1.f:0.f;
    return ret;
}

vec3 Extractor::computeDensity(ivec3 stu, float anglethreshold)
{
    std::vector<vec3> dirs;
    dirs.push_back(getNorm(df, w, stu+ivec3(-1,-1,-1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(-1,-1,0)));
    dirs.push_back(getNorm(df, w, stu+ivec3(-1,-1,1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(-1,0,-1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(-1,0,0)));
    dirs.push_back(getNorm(df, w, stu+ivec3(-1,0,1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(-1,1,-1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(-1,1,0)));
    dirs.push_back(getNorm(df, w, stu+ivec3(-1,1,1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(0,-1,-1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(0,-1,0)));
    dirs.push_back(getNorm(df, w, stu+ivec3(0,-1,1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(0,0,-1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(0,0,0)));
    dirs.push_back(getNorm(df, w, stu+ivec3(0,0,1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(0,1,-1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(0,1,0)));
    dirs.push_back(getNorm(df, w, stu+ivec3(0,1,1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(1,-1,-1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(1,-1,0)));
    dirs.push_back(getNorm(df, w, stu+ivec3(1,-1,1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(1,0,-1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(1,0,0)));
    dirs.push_back(getNorm(df, w, stu+ivec3(1,0,1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(1,1,-1)));
    dirs.push_back(getNorm(df, w, stu+ivec3(1,1,0)));
    dirs.push_back(getNorm(df, w, stu+ivec3(1,1,1)));
    float smallangles = 0;
    float largeangles = 0;
    int numsmallangles = 0;
    int numlargeangles = 0;
    for (int i = 0; i < neighbors.size(); i+=2) {
        float a = angle(dirs[neighbors[i]], dirs[neighbors[i+1]]);
        if (a > anglethreshold) {
            largeangles += a;
            numlargeangles++;
        } else {
            smallangles += a;
            numsmallangles++;
        }
    }
    return vec3(numsmallangles?smallangles/numsmallangles:0, numsmallangles, numlargeangles?largeangles/numlargeangles:0);
}

void Extractor::extract(std::vector<float>& points, float threshold)
{
    float* densitymap = new float[w*w*w];
    memset(densitymap, 0, sizeof(float)*w*w*w);
    std::mutex vectormutex;

#pragma omp parallel for
    for (int i = 1; i < w-1; i++) {
        for (int j = 1; j < w-1; j++) {
            for (int k = 1; k < w-1; k++) {
                //float f = medialaxis(df, w, ivec3(i,j,k));
                vec3 d = computeDensity(ivec3(i,j,k), threshold);
                float f = neighbors.size()/2 - d.y;
                if (f > 8) {
                    densitymap[w*w*i + w*j + k] = d.x;
                } else {
                    densitymap[w*w*i + w*j + k] = -1;
                }
            }
        }
    }
#pragma omp parallel for
    for (int i = 1; i < w-1; i++) {
        for (int j = 1; j < w-1; j++) {
            for (int k = 1; k < w-1; k++) {
                int idx = i*w*w+j*w+k;
                if (densitymap[idx] >= 0) {
                    std::lock_guard<std::mutex> lock(vectormutex);
                    points.push_back((k+0.5)/(float)w);
                    points.push_back((j+0.5)/(float)w);
                    points.push_back((i+0.5)/(float)w);
                    points.push_back(densitymap[idx]);
                    points.push_back(densitymap[idx]);
                    points.push_back(densitymap[idx]);
                }
            }
        }
    }
}

void Extractor::initNeighbors() {
    std::vector<float> tmp;
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                tmp.push_back(9*r + 3*i + j);
                tmp.push_back(9*r + 3*i + j+1);
                tmp.push_back(9*r + 3*j + i);
                tmp.push_back(9*r + 3*(j+1) + i);
                tmp.push_back(9*j + 3*r + i);
                tmp.push_back(9*(j+1) + 3*r + i);
            }
        }
    }
    int MIDDLE = 3*3 + 3 + 1;
    for (int i = 0; i < tmp.size(); i+=2) {
        if (tmp[i] != MIDDLE && tmp[i+1] != MIDDLE) {
            neighbors.push_back(tmp[i]);
            neighbors.push_back(tmp[i+1]);
        }
    }
}

