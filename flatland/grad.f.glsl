#version 330
in vec2 st;
out vec4 color;
uniform sampler2D buffer;
uniform sampler2D aux;
uniform ivec2 dim;
uniform float exposure;

float gradient(vec2 st) {
    float a = 1.f/dim.x;
    float dx = texture(buffer, st+vec2(a,-a)).x
             + texture(buffer, st+vec2(a,0)).x
             + texture(buffer, st+vec2(a,a)).x
             - texture(buffer, st+vec2(-a,-a)).x
             - texture(buffer, st+vec2(-a,0)).x
             - texture(buffer, st+vec2(-a,a)).x;
    float dy = texture(buffer, st+vec2(-a,a)).x
             + texture(buffer, st+vec2(0,a)).x
             + texture(buffer, st+vec2(a,a)).x
             - texture(buffer, st+vec2(-a,-a)).x
             - texture(buffer, st+vec2(0,-a)).x
             - texture(buffer, st+vec2(a,-a)).x;
    return sqrt(dx*dx + dy*dy);
}

void main() {
    // float v = texture(buffer, st).x*exposure;
    float v = gradient(st)*exposure;
    float a = texture(aux, st).x;
    vec4 w = a>0.5?vec4(a,0,0,0):vec4(0,0.5+a,0,0);
    color = a>0?w:vec4(v,v,v,1);
};
