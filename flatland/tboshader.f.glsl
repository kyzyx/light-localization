#version 330
in vec2 st;
out vec4 color;
uniform sampler2D buffer;
uniform sampler2D aux;
uniform ivec2 dim;
uniform float exposure;

const int NUM_KELLY_COLORS = 20;
uniform vec4 KellyColors[NUM_KELLY_COLORS] = vec4[NUM_KELLY_COLORS](
    vec4(255, 179, 0, 255)/255,
    vec4(128, 62, 117, 255)/255,
    vec4(255, 104, 0, 255)/255,
    vec4(166, 189, 215, 255)/255,
    vec4(193, 0, 32, 255)/255,
    vec4(206, 162, 98, 255)/255,
    vec4(129, 112, 102, 255)/255,
    vec4(0, 125, 52, 255)/255,
    vec4(246, 118, 142, 255)/255,
    vec4(0, 83, 138, 255)/255,
    vec4(255, 122, 92, 255)/255,
    vec4(83, 55, 122, 255)/255,
    vec4(255, 142, 0, 255)/255,
    vec4(179, 40, 81, 255)/255,
    vec4(244, 200, 0, 255)/255,
    vec4(127, 24, 13, 255)/255,
    vec4(147, 170, 0, 255)/255,
    vec4(89, 51, 21, 255)/255,
    vec4(241, 58, 19, 255)/255,
    vec4(35, 44, 22, 255)/255
);

void main() {
    float v = texture(buffer, st).x*exposure;
    int i = int(texture(aux, st).x*10);
    color = i>0?KellyColors[i-1]:vec4(v,v,v,1);
};
