#version 330
in vec3 stu;
out vec4 color;
uniform sampler3D buffer;

const float PI = 3.1415926536;
const float MAX_FLOAT = 1e9;

void main() {
    vec4 m = texture(buffer, stu);
    int ispherical = floatBitsToInt(m.y);
    int itheta = ispherical&((1<<15)-1);
    int iphi = ispherical>>15;
    float theta = itheta*PI/(1<<15);
    float phi = iphi*2*PI/(1<<15);
    vec3 n = vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    color = m.x>=MAX_FLOAT?vec4(0,0,0,1):0.5*vec4(n, 1)+0.5;
};
