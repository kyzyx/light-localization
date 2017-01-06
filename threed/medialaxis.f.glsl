#version 330
in vec3 stu;
out vec4 color;
uniform sampler3D buffer;
uniform ivec2 dim;
uniform float exposure;
uniform float threshold;

const float PI = 3.1415926536;

float angle(vec3 a, vec3 b) {
    return atan(length(cross(a,b))/dot(a,b));
}

vec3 getNorm(vec3 c) {
    vec4 m = texture(buffer, c);
    int ispherical = floatBitsToInt(m.y);
    int itheta = ispherical&((1<<15)-1);
    int iphi = ispherical>>15;
    float theta = itheta*PI/(1<<15);
    float phi = iphi*2*PI/(1<<15);
    return vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
}

float medialaxis(vec3 stu) {
    float a = 1.f/dim.x;
    float ret = 0;
    vec3 v = getNorm(stu);
    ret += angle(v, getNorm(stu+vec3(-a,-a,-a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(-a,-a,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(-a,-a,a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(-a,0,-a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(-a,0,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(-a,0,a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(-a,a,-a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(-a,a,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(-a,a,a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(0,-a,-a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(0,-a,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(0,-a,a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(0,0,-a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(0,0,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(0,0,a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(0,a,-a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(0,a,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(0,a,a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(a,-a,-a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(a,-a,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(a,-a,a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(a,0,-a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(a,0,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(a,0,a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(a,a,-a)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(a,a,0)))>threshold?1.f:0.f;
    ret += angle(v, getNorm(stu+vec3(a,a,a)))>threshold?1.f:0.f;
    return ret/27.f;
}

void main() {
    float v = medialaxis(stu)*exposure;
    color = vec4(v,v,v,1);
};
