#version 330
layout (location=0) in vec4 pos;
uniform mat4 modelviewmatrix;
uniform mat4 projectionmatrix;
out vec3 vcolor;

void main() {
   gl_Position = projectionmatrix*modelviewmatrix*vec4(pos.xyz,1);
   float f = 30*pos.w;
   vcolor = vec3(0, f, f);
}
