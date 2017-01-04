#version 330
layout (location=0) in vec3 pos;
layout (location=1) in vec2 coord;
uniform vec3 pt;
uniform vec3 xax;
uniform vec3 yax;
uniform mat4 modelviewmatrix;
uniform mat4 projectionmatrix;
out vec2 st;

void main() {
   st = coord.st;
   vec3 c = (1+pos)/2;
   vec3 pp = pt + xax*c.x + yax*c.y;
   gl_Position = projectionmatrix*modelviewmatrix*vec4(pp,1);
}
