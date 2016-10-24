#version 330
in vec2 st;
out vec4 color;
uniform sampler2D buffer;
uniform sampler2D aux;
uniform ivec2 dim;
uniform float exposure;
const int NUM_KELLY_COLORS = 20;
uniform vec4 KellyColors[NUM_KELLY_COLORS] = vec4[NUM_KELLY_COLORS](
    vec4(255, 179, 0, 128)/255,
    vec4(128, 62, 117, 128)/255,
    vec4(255, 104, 0, 128)/255,
    vec4(166, 189, 215, 128)/255,
    vec4(193, 0, 32, 128)/255,
    vec4(206, 162, 98, 128)/255,
    vec4(129, 112, 102, 128)/255,
    vec4(0, 125, 52, 128)/255,
    vec4(246, 118, 142, 128)/255,
    vec4(0, 83, 138, 128)/255,
    vec4(255, 122, 92, 128)/255,
    vec4(83, 55, 122, 128)/255,
    vec4(255, 142, 0, 128)/255,
    vec4(179, 40, 81, 128)/255,
    vec4(244, 200, 0, 128)/255,
    vec4(127, 24, 13, 128)/255,
    vec4(147, 170, 0, 128)/255,
    vec4(89, 51, 21, 128)/255,
    vec4(241, 58, 19, 128)/255,
    vec4(35, 44, 22, 128)/255
);
void main() {
    int idx = floatBitsToInt(texture(buffer, st).y)%NUM_KELLY_COLORS;
    float v = texture(buffer, st).x*exposure;
    float a = texture(aux, st).x;
    vec4 w = a>0.5?vec4(a,0,0,0):vec4(0,0.5+a,0,0);
    color = a>0?w:KellyColors[idx];
};
