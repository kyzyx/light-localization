import sys

dim = 3
skip = 1

if len(sys.argv) > 1:
    dim = int(sys.argv[1])
    if len(sys.argv) > 2:
        skip = int(sys.argv[2])

x = []
y = []
for i in range(-dim, dim, skip):
    x.append(-dim)
    y.append(i)
for i in range(-dim, dim, skip):
    x.append(i)
    y.append(dim)
for i in range(dim, -dim, -skip):
    x.append(dim)
    y.append(i)
for i in range(dim, -dim, -skip):
    x.append(i)
    y.append(-dim)
x.append(x[0])
y.append(y[0])

# Print out vec
shadertext1 = """#version 330
in vec2 st;
out vec4 color;
uniform sampler2D buffer;
uniform sampler2D aux;
uniform ivec2 dim;
uniform float exposure;
uniform int threshold;
uniform int maxidx;

"""

shadertext2 = """const int NUM_KELLY_COLORS = 20;
uniform vec4 KellyColors[NUM_KELLY_COLORS] = vec4[NUM_KELLY_COLORS](
    vec4(255, 179, 0, 255)/255,
    vec4(128, 62, 117, 255)/255,
    vec4(255, 104, 0, 255)/255,
    vec4(166, 189, 215, 255)/255,
    vec4(193, 0, 32, 255)/255,
    vec4(206, 162, 98, 255)/255,
    vec4(129, 112, 102, 255)/255,
    vec4(0, 125, 52, 255)/255,
    vec4(246, 118, 142, 255)/255,
    vec4(0, 83, 138, 255)/255,
    vec4(255, 122, 92, 255)/255,
    vec4(83, 55, 122, 255)/255,
    vec4(255, 142, 0, 255)/255,
    vec4(179, 40, 81, 255)/255,
    vec4(244, 200, 0, 255)/255,
    vec4(127, 24, 13, 255)/255,
    vec4(147, 170, 0, 255)/255,
    vec4(89, 51, 21, 255)/255,
    vec4(241, 58, 19, 255)/255,
    vec4(35, 44, 22, 255)/255
);

int cdist(int a, int b) {
    int d = max(b-a,0);
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
    int i = int(texture(aux, st).x*10);
    color = i>0?KellyColors[i-1]:vec4(v,v,v,1);
};
"""

f = open("density.f.glsl", "w")
f.write(shadertext1)
f.write("const int NUM_ADJ = %d;\n"%(len(x)-1))
f.write("uniform vec2 adj[%d] = vec2[%d](\n"%(len(x), len(x)))
vec = ["vec2(%d,%d)"%(x[i], y[i]) for i in range(len(x))]
f.write(",\n".join(vec))
f.write("\n);\n")
f.write(shadertext2)

# Print out arrays
f = open("adj.gen.h", "w")
f.write("const int NUM_ADJ = %d;\n"%(len(x)))
f.write("__constant__ int adjx[NUM_ADJ] = {%s};\n"%(",".join([str(i) for i in x])))
f.write("__constant__ int adjy[NUM_ADJ] = {%s};\n"%(",".join([str(i) for i in y])))
