#version 330
in vec2 st;
out vec4 color;
uniform sampler2D buffer;
uniform sampler2D aux;
uniform ivec2 dim;
uniform float exposure;
uniform int threshold;
uniform int maxidx;

int cdist(int a, int b) {
    int d = abs(a-b);
    return d>maxidx/2?maxidx-d:d;
}

float medialaxis(vec2 st) {
    float a = 1.f/dim.x;
    int v = floatBitsToInt(texture(buffer, st).y);
    float ret = 0;
    ret += cdist(v, floatBitsToInt(texture(buffer, st+vec2(-a,-a)).y))>threshold?1.f:0.f;
    ret += cdist(v, floatBitsToInt(texture(buffer, st+vec2(-a,0)).y))>threshold?1.f:0.f;
    ret += cdist(v, floatBitsToInt(texture(buffer, st+vec2(-a,a)).y))>threshold?1.f:0.f;
    ret += cdist(v, floatBitsToInt(texture(buffer, st+vec2(0,-a)).y))>threshold?1.f:0.f;
    ret += cdist(v, floatBitsToInt(texture(buffer, st+vec2(0,0)).y))>threshold?1.f:0.f;
    ret += cdist(v, floatBitsToInt(texture(buffer, st+vec2(0,a)).y))>threshold?1.f:0.f;
    ret += cdist(v, floatBitsToInt(texture(buffer, st+vec2(a,-a)).y))>threshold?1.f:0.f;
    ret += cdist(v, floatBitsToInt(texture(buffer, st+vec2(a,0)).y))>threshold?1.f:0.f;
    ret += cdist(v, floatBitsToInt(texture(buffer, st+vec2(a,a)).y))>threshold?1.f:0.f;
    return ret;
}

void main() {
    float v = medialaxis(st)*exposure;
    float a = texture(aux, st).x;
    vec4 w = a>0.5?vec4(a,0,0,0):vec4(0,0.5+a,0,0);
    color = a>0?w:vec4(v,v,v,1);
};
