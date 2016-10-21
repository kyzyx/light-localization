#version 330
in vec2 st;
out vec4 color;
uniform sampler2D buffer;
uniform sampler2D aux;
uniform ivec2 dim;
uniform float exposure;

float localmin(vec2 st) {
    float a = 1.f/dim.x;
    float ov = texture(buffer, st).x; 
    float v = ov;
    v = min(v, texture(buffer, st+vec2(a,-a)).x);
    v = min(v, texture(buffer, st+vec2(a,0)).x);
    v = min(v, texture(buffer, st+vec2(a,a)).x);
    v = min(v, texture(buffer, st+vec2(0,-a)).x);
    v = min(v, texture(buffer, st+vec2(0,0)).x);
    v = min(v, texture(buffer, st+vec2(0,a)).x);
    v = min(v, texture(buffer, st+vec2(-a,-a)).x);
    v = min(v, texture(buffer, st+vec2(-a,0)).x);
    v = min(v, texture(buffer, st+vec2(-a,a)).x);
    return ov-v;
}

void main() {
    float v = localmin(st)>0?0:1;
    float a = texture(aux, st).x;
    vec4 w = a>0.5?vec4(a,0,0,0):vec4(0,0.5+a,0,0);
    color = a>0?w:vec4(v,v,v,1);
};
