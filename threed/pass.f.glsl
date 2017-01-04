#version 330
in vec3 vcolor;
in vec3 vnormal;
out vec4 color;

void main() {
    color = vec4(vcolor,1);
}
