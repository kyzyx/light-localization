#version 330
in vec2 st;
out vec4 color;
uniform samplerBuffer buffer;
uniform ivec2 dim;
uniform float exposure;

void main() {
    int i = int(st.x * float(dim.x));
    int j = int(st.y * float(dim.y));
    float v = texelFetch(buffer, i+dim.x*j).x*exposure;
    color = vec4(v,v,v,1);
};
