#version 330
layout (location=0) in vec3 pos;
layout (location=1) in vec2 coord;
uniform vec3 pt;
uniform vec3 xax;
uniform vec3 yax;
out vec3 stu;

void main() {
   vec3 c = (1+pos)/2;
   stu = pt + xax*c.x + yax*c.y;
   gl_Position = vec4(pos,1);
}
