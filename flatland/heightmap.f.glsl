#version 330
in vec3 normal;
out vec4 color;

void main() {
    // float v = clamp(dot( normal, normalize(vec3(1,1,0))),0,1);
    float v = abs(dot(normal, normalize(vec3(0,0,1))));
    // color = vec4((normal+1)/2,1);
    color = vec4(v,v,v,1);
}
