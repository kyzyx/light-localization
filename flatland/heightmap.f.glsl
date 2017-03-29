#version 330
in vec3 normal;
in vec2 st;
out vec4 color;

void main() {
    float v = abs(dot(normal, normalize(vec3(1,1,0.1))));
    ivec2 t = ivec2(int(st.x*10), int(st.y*10));
    vec3 diffuse = ((t.x%2)+(t.y%2))%2 == 0?vec3(1,0,0):vec3(1,1,1);
    vec3 ambient = vec3(0.1, 0.1, 0.1);
    // color = vec4((normal+1)/2,1);
    color = vec4(v*diffuse + ambient,1);
}
