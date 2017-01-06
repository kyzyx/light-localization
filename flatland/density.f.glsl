#version 330
in vec2 st;
out vec4 color;
uniform sampler2D buffer;
uniform sampler2D aux;
uniform ivec2 dim;
uniform float exposure;
uniform int threshold;
uniform int maxidx;

const int NUM_ADJ = 8;
uniform vec2 adj[9] = vec2[9](
    vec2(-1,-1),
    vec2(-1,0),
    vec2(-1,1),
    vec2(0,1),
    vec2(1,1),
    vec2(1,0),
    vec2(1,-1),
    vec2(0,-1),
    vec2(-1,-1)
);

int cdist(int a, int b) {
    int d = abs(a-b);
    return d>maxidx/2?maxidx-d:d;
}

float medialaxis(vec2 st) {
    float a = 1.f/dim.x;
    float ret = 0;
    int count = 0;
    int prev = floatBitsToInt(texture(buffer, st+a*adj[0]).y);
    for (int i = 0; i < NUM_ADJ; i++) {
        int curr = floatBitsToInt(texture(buffer, st+a*adj[i+1]).y);
        int d = cdist(prev, curr);
        count += d>threshold?0:1;
        ret += d>threshold?0:d;
        prev = curr;
    }
    return 0.5*ret/count;
}

void main() {
    float v = medialaxis(st)*exposure;
    float a = texture(aux, st).x;
    vec4 w = a>0.5?vec4(a,0,0,0):vec4(0,0.5+a,0,0);
    color = a>0?w:vec4(v,v,v,1);
};
