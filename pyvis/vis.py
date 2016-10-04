import sys,math
from visual import *

def readPly(filename):
    lines = open(filename,'r').readlines()
    verts = []
    retv = []
    retn = []
    retc = []
    normals = []
    colors = []
    nverts = 0
    nfaces = 0
    state = 0

    hn = lines.index("end_header\n")+1
    header = lines[:hn]
    for line in header:
        if line.strip() == "end_header":
            state = 1
        elif line.split()[0] == "element":
            if line.split()[1] == "vertex":
                nverts = int(line.split()[2])
            elif line.split()[1] == "face":
                nfaces = int(line.split()[2])
    for line in lines[hn:hn+nverts]:
        l = [float(x) for x in line.split()]
        verts.append((l[0],l[1],l[2]))
        normals.append(vector(l[3],l[4],l[5]).norm())
        colors.append(vector(l[6],l[7],l[8])/255.)
    for line in lines[hn+nverts:]:
        retv.extend([verts[int(i)] for i in line.split()[1:4]])
        retn.extend([normals[int(i)] for i in line.split()[1:4]])
        retc.extend([colors[int(i)] for i in line.split()[1:4]])
    return retv,retn,retc

if len(sys.argv) < 4:
    print "Usage: python vis.py filename.ply solution.txt spacing"
else:
    filename = sys.argv[1]
    v,n,c = readPly(filename)
    mesh = faces(pos=v,color=c)
    mesh = faces(pos=v,normal=n,color=c)

    #bbmin = [-0.5,-0.5,-0.5]
    #bbmax = [0.5,0.5,0.5]
    bbmin = [min([x[i] for x in mesh.pos]) for i in range(3)]
    bbmax = [max([x[i] for x in mesh.pos]) for i in range(3)]

    spacing = float(sys.argv[3])
    bbl = [bbmax[i]-bbmin[i] for i in range(3)]
    bbls = [int(l/spacing - 1) for l in bbl]
    bbs = [bbmin[i] + (bbl[i] - bbls[i]*spacing)*0.5 for i in range(3)]

    fn = sys.argv[2]

    data = [[float(x) for x in l.split()] for l in open(fn,"r").readlines()]
    directions = []
    for i in range(3):
        for s1 in range(-1,2,2):
            for s2 in range(-1,2,2):
                v = vector(0,0,0)
                v[i] = s1
                v[(i+1)%3] = 1.618034*s2
                directions.append(v.norm())

    idx = 0
    lines = []
    nonzero = []
    l = 0.03
    dmax = max([d[0] for d in data if d[0] > 0])
    dmin = min([d[0] for d in data if d[0] > 0])
    dr = math.log(dmax/dmin)
    if dmax == dmin:
        dr = 1
    l = l/dr;

    for a in range(bbls[0]):
        for b in range(bbls[1]):
            for c in range(bbls[2]):
                if sum(data[idx]) > 0:
                    p = vector(0,0,0)
                    p[0] = bbs[0] + a*spacing
                    p[1] = bbs[1] + b*spacing
                    p[2] = bbs[2] + c*spacing
                    #m = log(data[idx][0]/dmin)
                    m = 20
                    lines.append(sphere(pos=p,radius=m*l))
                    nonzero.append(idx)
                idx = idx+1

    #for a in range(bbls[0]):
        #for b in range(bbls[1]):
            #for c in range(bbls[2]):
                #for d in range(len(directions)):
                    #if sum(data[idx]) > 0:
                        #p = vector(0,0,0)
                        #p[0] = bbs[0] + a*spacing
                        #p[1] = bbs[1] + b*spacing
                        #p[2] = bbs[2] + c*spacing
                        #m = log(data[idx][0]/dmax)
                        #lines.append(curve(pos=[p,p+l*m*directions[d]]))
                        #nonzero.append(idx)
                    #idx = idx+1

    #l1 = sphere(pos=(0,0,0),radius=0.03,color=(1,0,0))
    #l1 = sphere(pos=(0,0,0.25),radius=0.03,color=(1,0,0))
    #l2 = cone(pos=(0,0.03,0),axis=(0,-0.03,0),radius=0.01, color=(1,0,0))

    def setVis(j):
        for i,idx in enumerate(nonzero):
            lines[i].visible = data[idx][j] > 0

    setVis(len(data[0])-1)
