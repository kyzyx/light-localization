from random import uniform

nlights = [2,3,4,5]
sep = 0.1
ncases = 100
filename = "box%03d.lt"

scene = [
"-1 -1.01 -1 1.01",
"-1.01 1 1.01 1",
"1 1.01 1 -1.01 ",
"1.01 -1 -1.01 -1"
]

def dist2(a,b):
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])

def genlights(n):
    lights = []
    valid = False
    while not valid:
        lights = [(uniform(-0.9,0.9), uniform(-0.9,0.9), uniform(1,5)) for i in range(n)]
        valid = True
        for i in range(len(lights)):
            for j in range(i+1,len(lights)):
                if dist2(lights[i], lights[j]) < sep*sep:
                    valid = False
    return ["%f %f %f"%l for l in lights]


for n in nlights:
    for i in range(ncases):
        f = open(filename%(n*ncases+i), "w")
        f.write("4 %d\n"%n);
        f.writelines('\n'.join(scene))
        f.write("\n")
        f.writelines('\n'.join(genlights(n)))
