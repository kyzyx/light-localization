#version 330
uniform sampler2D depth;
uniform int dim;
uniform float foc;
in vec3 vcolor;
in vec3 vnormal;
out vec4 color;

vec3 v(vec2 st) {
    return vec3((st.x-dim/2+0.5)/foc,(st.y-dim/2+0.5)/foc,1);
}

vec3 compute_normal(vec2 st) {
    vec2 xy = st/dim;
    float a = 1.f/dim;
    float z = texture(depth, xy).x;
    float zx = texture(depth, xy+vec2(2*a,0)).x;
    float zy = texture(depth, xy+vec2(0,2*a)).x;
    vec3 p  = z*v(st);
    vec3 dx = (zx*v(st+vec2(2,0)) - p);
    vec3 dy = (zy*v(st+vec2(0,2)) - p);
    return normalize(cross(dx,dy));
}

vec3 light(vec3 n) {
    return abs(dot(n,vec3(0,0,1))) + vec3(0.1,0.1,0.1);
}

void main() {
    vec3 n = compute_normal(gl_FragCoord.xy);
    color = vec4(vcolor*light(n),1);
}
