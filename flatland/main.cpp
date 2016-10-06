#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include "cudamap.h"

using namespace std;
using namespace Eigen;

class Line {
    public:
        Line()
            : p1(Vector2f(0,0)), p2(Vector2f(0,0)) {}
        Line(Vector2f a, Vector2f b)
            : p1(a), p2(b) {}
        Vector2f p1;
        Vector2f p2;

        Vector2f vec() const {
            return (p2-p1).normalized();
        }
        Vector2f normal() const {
            Vector2f v = (p2-p1).normalized();
            return Vector2f(v[1], -v[0]);
        }
};

float EPSILON = 1e-5;
float ccw(Vector2f a, Vector2f b, Vector2f c) {
    return (b[0] - a[0])*(c[1] - a[1]) - (c[0] - a[0])*(b[1] - a[1]);
}
bool intersects(Line l1, Line l2) {
    if (ccw(l1.p1, l1.p2, l2.p1)*ccw(l1.p1, l1.p2, l2.p2) > 0) return false;
    if (ccw(l2.p1, l2.p2, l1.p1)*ccw(l2.p1, l2.p2, l1.p2) > 0) return false;
    return true;
}
Vector2f projectPointToLine(Vector2f point, Line line) {
    Vector2f v = line.vec();
    return line.p1 + (point-line.p1).dot(v)*v;
}

class Scene {
    public:
        Scene() : minp(Vector2f(0,0)), maxp(Vector2f(0,0)) { }
        void addSegment(Line l) {
            extendBbox(l.p1);
            extendBbox(l.p2);
            scene.push_back(l);
        }

        float lightPixel(Vector2f p, Vector2f n) {
            float ret = 0;
            for (int i = 0; i < lights.size(); i++) {
                Vector2f L = lights[i].head(2) - p;
                Vector2f p2 = p + L.normalized()*0.0001;
                if (L.squaredNorm() == 0) continue;
                ret += fabs(L.normalized().dot(n))*lights[i][2]/L.squaredNorm();
                //if (!intersectsAny(Line(p2, lights[i].head(2)))) {
                    //ret += lights[i][2]/L.squaredNorm();
                    ////cout << i << " " << lights[i][2] << " " << p[0] << " " << p[1] << " " << L[0] << " " << L[1] << " " << L.squaredNorm() << " " << ret << endl;
                //}
            }
            return ret;
        }

        int intersectsAny(Line a) {
            for (int i = 0; i < scene.size(); i++) {
                if (intersects(scene[i],a)) {
                    return i+1;
                }
            }
            return 0;
        }

        void traceLine(int w, int h, Line l) {
            int x0, y0, x1, y1;
            world2clip(l.p1, x0, y0, w, h);
            world2clip(l.p2, x1, y1, w, h);

            int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
            int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1;
            int err = (dx>dy ? dx : -dy)/2;
            int e2;

            while(1) {
                Vector2f n = l.normal();
                Vector2f p = projectPointToLine(clip2world(x0, y0, w, h),l);
                coords.push_back(x0);
                coords.push_back(y0);
                surfels.push_back(p[0]);
                surfels.push_back(p[1]);
                surfels.push_back(n[0]);
                surfels.push_back(n[1]);
                intensities.push_back(0);
                if (x0==x1 && y0==y1) break;
                e2 = err;
                if (e2 >-dx) { err -= dy; x0 += sx; }
                if (e2 < dy) { err += dx; y0 += sy; }
            }
        }

        void initCuda(int w, int h) {
            cm.w = w;
            cm.h = h;
            cm.maxx = maxp[0];
            cm.maxy = maxp[1];
            cm.minx = minp[0];
            cm.miny = minp[1];
            cm.n = surfels.size()/4;
            Cudamap_init(&cm, surfels.data());
        }
        void lightScene(float* img, int w, int h) {
            memset(img, 0, 3*w*h*sizeof(float));
            for (int i = 0; i < intensities.size(); i++) {
                int x = coords[2*i];
                int y = coords[2*i+1];
                Vector2f p(surfels[4*i+0], surfels[4*i+1]);
                Vector2f n(surfels[4*i+2], surfels[4*i+3]);
                intensities[i] = lightPixel(p, n);
                img[3*(x + w*y)] = intensities[i];
                img[3*(x + w*y)+1] = n[0];
                img[3*(x + w*y)+2] = n[1];
            }
        }

        void render(int w, int h) {
            for (int i = 0; i < scene.size(); i++) {
                traceLine(w, h, scene[i]);
            }
        }

        void computeFieldCuda(float* v, float* field, int w, int h) {
            initCuda(w,h);
            Cudamap_setIntensities(&cm, intensities.data());
            Cudamap_compute(&cm, field);
            Cudamap_free(&cm);

            for (int i = 0; i < w*h; i++) {
                if (v[3*i] > 0) field[i] = -1;
            }
            for (int i = 0; i < lights.size(); i++) {
                int x, y;
                world2clip(lights[i].head(2), x, y, w, h);
                field[x+y*w] = -2;
            }
        }
        void computeField(float* v, float* field, int w, int h) {
            Vector2f bounds = maxp - minp;
            float d = fmax(bounds[0]/(w-2), bounds[1]/(h-2));
            for (int i = 0; i < w*h; i++) field[i] = 1e7;
            for (int i = 0; i < w*h; i++) {
                if (v[3*i] > 0) field[i] = -1;
                else continue;
                Vector2f p = clip2world(i%w,i/w,w,h);
                for (int j = 0; j < w*h; j++) {
                    if (v[3*j] > 0) continue;
                    Vector2f p2 = clip2world(j%w,j/w,w,h);
                    Vector2f L = p2 - p;
                    Vector2f Ln = L.normalized();
                    Line l(p,p2);
                    l.p1 += Ln*d;
                    l.p2 -= Ln*d;
                    Vector2f n(v[3*i+1], v[3*i+2]);
                    //if (intersectsAny(l)) continue;
                    if (n.dot(Ln) < 1e-9) continue;
                    field[j] = min(field[j], v[3*i]*L.squaredNorm()/n.dot(Ln));
                }
            }
            for (int i = 0; i < lights.size(); i++) {
                int x, y;
                world2clip(lights[i].head(2), x, y, w, h);
                field[x+y*w] = -2;
            }
        }

        vector<Vector3f> lights;
    protected:
        Vector2f clip2world(int x, int y, int w, int h) {
            Vector2f v = maxp - minp;
            v[0] *= (x+1.5)/((float)w-2);
            v[1] *= (y+1.5)/((float)h-2);
            return v + minp;
        }
        void world2clip(Vector2f p, int& x, int& y, int w, int h) {
            Vector2f v = maxp - minp;
            Vector2f d = p - minp;
            x = d[0]/v[0]*(w-2) + 1;
            y = d[1]/v[1]*(h-2) + 1;
        }

        void extendBbox(Vector2f p) {
            minp[0] = min(minp[0], p[0]);
            minp[1] = min(minp[1], p[1]);
            maxp[0] = max(maxp[0], p[0]);
            maxp[1] = max(maxp[1], p[1]);
        }
        Vector2f minp, maxp;
        vector<Line> scene;

        // Intermediates
        Cudamap cm;
        vector<float> surfels;
        vector<int> coords;
        vector<float> intensities;
};


void saveFile(const char* filename, float* img, int w, int h, int ch=1) {
    int m = 65535;
    ofstream out(filename);
    out << "P3 " << w << " " << h << " " << m << endl;
    int l = 16383;
    for (int y = h-1; y >= 0; y--) {
        for (int x = 0; x < w; x++) {
            int idx = x+y*w;
            if (img[ch*idx] < 0) {
                if (img[ch*idx] < -1) {
                    out << (int) -l*img[ch*idx] << " 0 0 ";
                } else {
                    out << "0 0 " << (int) -l*img[ch*idx] << " ";
                }
            } else if (l*img[ch*idx] >= m) {
                out << "0 " <<  l << " 0 ";
            } else {
                out << (int) (l*img[ch*idx]) << " " << (int) (l*img[ch*idx]) << " " << (int) (l*img[ch*idx]) << " ";
            }
        }
        out << endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: flatland width height filename [-cuda]" << endl;
        return 0;
    }
    Scene s;

    // Read in scene
    int nsegs, nlights;
    double x,y,z;
    cin >> nsegs >> nlights;
    for (int i = 0; i < nsegs; i++) {
        cin >> x >> y;
        Vector2f v1(x,y);
        cin >> x >> y;
        Vector2f v2(x,y);
        s.addSegment(Line(v1,v2));
    }
    for (int i = 0; i < nlights; i++) {
        cin >> x >> y >> z;
        s.lights.push_back(Vector3f(x,y,z));
    }
    // Render scene
    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    float* img = new float[3*w*h];
    s.render(w, h);
    s.lightScene(img, w, h);
    if (argc > 3) saveFile("img.ppm", img, w, h, 3);

    // Compute field
    float* field = new float[w*h];
    if (argc > 4 && strcmp(argv[4], "-cuda") == 0) {
        s.computeFieldCuda(img, field, w, h);
    } else {
        s.computeField(img, field, w, h);
    }
    if (argc > 3) saveFile(argv[3], field, w, h);
}
