from random import uniform

nlights = [2,3,4,5]
sep = 0.1
ncases = 100
filename = "cube%02d.ply"


def dist2(a,b):
    return sum([(a[i]-b[i])*(a[i]-b[i]) for i in range(3)])

def genlights(n):
    lights = []
    valid = False
    while not valid:
        lights = [(uniform(0.1,0.9), uniform(0.1,0.9), uniform(0.1,0.9), uniform(1,5)) for i in range(n)]
        valid = True
        for i in range(len(lights)):
            for j in range(i+1,len(lights)):
                if dist2(lights[i], lights[j]) < sep*sep:
                    valid = False
    return ["%f %f %f %f"%l for l in lights]


lines = open("cube40.ply", "r").readlines()
idx = lines.index("element light 2\n")
if idx < 0:
    print "Error ", idx
else:
    header = lines[:idx]
    remainder = lines[idx+1:]
    for n in nlights:
        for i in range(ncases):
            f = open(filename%(n*ncases+i), "w")
            f.writelines(header)
            f.write("element light %d\n"%n);
            f.writelines(remainder)
            f.writelines('\n'.join(genlights(n)))
