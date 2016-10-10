#version 330
in vec2 st;
out vec4 color;
uniform sampler2D buffer;
uniform ivec2 dim;
uniform float exposure;

void main() {
    float v = texture(buffer, st).x*exposure;
    color = vec4(v,v,v,1);
};
