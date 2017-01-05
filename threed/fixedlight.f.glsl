#version 330
in vec3 vcolor;
in vec3 vnormal;
out vec4 color;

void main() {
    float v = clamp(dot(vnormal,vec3(0,1,0)),0,1);
    color = vec4(v,v*0.1,v*0.1,1);
}
