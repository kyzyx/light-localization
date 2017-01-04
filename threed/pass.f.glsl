#version 330
in vec3 vcolor;
in vec3 vnormal;
uniform float exposure;
out vec4 color;

void main() {
    color = vec4(exposure*vcolor,1);
}
