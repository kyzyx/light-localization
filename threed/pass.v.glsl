#version 330
layout (location=0) in vec3 pos;
layout (location=1) in vec3 normal;
layout (location=2) in vec3 color;
uniform mat4 modelviewmatrix;
uniform mat4 projectionmatrix;
out vec3 vnormal;
out vec3 vcolor;

void main() {
   gl_Position = projectionmatrix*modelviewmatrix*vec4(pos,1);
   vnormal = normal;
   vcolor = color;
}
