#version 330
layout (location=0) in vec3 pos;
out vec3 normal;
out vec2 st;

uniform float exposure;
uniform mat4 view;
uniform mat4 proj;
uniform ivec2 dim;
uniform sampler2D buffer;

uniform const float EINF = 1e3;

vec3 gradient(vec2 st) {
    float a = 1.f/dim.x;
    float f0 = texture(buffer, st+vec2(0,a)).x;
    float f1 = texture(buffer, st+vec2(0,-a)).x;
    float f2 = texture(buffer, st+vec2(a,0)).x;
    float f3 = texture(buffer, st+vec2(-a,0)).x;
    f0 = f0>EINF?0:f0;
    f1 = f1>EINF?0:f1;
    f2 = f2>EINF?0:f2;
    f3 = f3>EINF?0:f3;
    vec3 dx = (vec3(2*a, 0, (f2-f3)*exposure));
    vec3 dy = (vec3(0, 2*a, (f0-f1)*exposure));
    return normalize(cross(dx,dy));
}

void main() {
   st = pos.xy+0.5+(0.5/dim.x);
   float f = texture(buffer, st).x;
   gl_Position = proj*view*vec4(pos.xy,f>EINF?0:f*exposure,1);
   normal = gradient(st);
}

