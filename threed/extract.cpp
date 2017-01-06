#include "extract.h"
#include <glm/glm.hpp>
#include <cmath>
#include <mutex>

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

const float threshold = M_PI/30.f;

float medialaxis(float* df, int dim, ivec3 stu) {
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

void extract(std::vector<float>& points, float* distancefield, int dim, float threshold)
{
    std::mutex vectormutex;
#pragma omp parallel for
    for (int i = 1; i < dim-1; i++) {
        for (int j = 1; j < dim-1; j++) {
            for (int k = 1; k < dim-1; k++) {
                float f = medialaxis(distancefield, dim, ivec3(i,j,k));
                if (f > threshold) {
                    std::lock_guard<std::mutex> lock(vectormutex);
                    points.push_back((k+0.5)/(float)dim);
                    points.push_back((j+0.5)/(float)dim);
                    points.push_back((i+0.5)/(float)dim);
                    points.push_back(f/27.f);
                }
            }
        }
    }

}
