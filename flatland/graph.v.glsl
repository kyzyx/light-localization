#version 330
layout (location=0) in float val;
uniform vec2 scale;
uniform int len;
uniform vec3 color;
out vec4 linecolor;

void main() {
    float x = gl_VertexID/float(len);
    linecolor = val>0?vec4(color,1):vec4(0.5*color,1);
    gl_Position = vec4(2*x-1, 2*abs(val)*scale.y-1, 0, 1);
}
