#version 330
in vec3 stu;
out vec4 color;
uniform sampler3D buffer;
uniform ivec2 dim;
uniform float exposure;

void main() {
    float v = texture(buffer, stu).x*exposure;
    //float a = texture(aux, st).x;
    //vec4 w = a>0.5?vec4(a,0,0,0):vec4(0,0.5+a,0,0);
    //color = a>0?w:vec4(v,v,v,1);
    color = vec4(v,v,v,1);
};
