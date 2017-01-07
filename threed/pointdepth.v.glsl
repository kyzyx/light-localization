#version 330
layout (location=0) in vec4 pos;
uniform mat4 modelviewmatrix;
uniform mat4 projectionmatrix;
out vec3 vcolor;

void main() {
   vec4 p = modelviewmatrix*vec4(pos.xyz,1);
   gl_Position = projectionmatrix*p;
   vcolor = vec3(-p.z, -p.z, -p.z);
}
