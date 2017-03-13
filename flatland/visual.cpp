#include "opengl_compat.h"
#include <vector>
#include <iostream>
#include <fstream>

#include "loadshader.h"
#include "geometry.h"
#include "cudamap.h"
#include "options.h"
#include "fileio.h"

const int EPSILON = 1e-9;

int displayscale = 4;
int width = 200;
int height = 200;
int lastCandidate = -1;
float prevErr = 1e9;
unsigned char* imagedata;
unsigned char* medialaxis;
float* imagecopy;
float* filtered;
float* distancefield;

using namespace std;
using namespace Eigen;

string helpstring =
"----- Visualization -----\n" \
"  [h] - display this help screen\n" \
"  [,] - decrease exposure\n" \
"  [.] - increase exposure\n" \
"  [m] - change visualization mode (RDF, RNF, Medial Axis)\n" \
"  [p] - print current scene description\n" \
"  [ ] - save image for current view\n" \
"----- Mouse -----\n" \
"  Clicking on an existing light selects it (red).\n" \
"  Clicking and dragging on an existing light moves it.\n" \
"  Clicking elsewhere creates a new uniform point light.\n" \
"  Shift-Clicking elsewhere creates a new directionally-varying light.\n" \
"----- Modify Selected Light -----\n" \
"  [[] - decrease selected light intensity\n" \
"  []] - increase selected light intensity\n" \
"  [Delete] - remove selected light\n" \
"----- Modify Selected Directionally-Varying Light -----\n" \
"  [q] - rotate light clockwise\n" \
"  [w] - rotate light counterclockwise\n" \
"----- Add Symmetrical Uniform Point Lights -----\n" \
"  [q] - add light horizontally symmetric with selected light\n" \
"  [w] - add light vertically symmetric with selected light\n" \
"  [e] - add 3 lights horizontally and vertically symmetric with selected light\n";

class Scene {
    public:
        Scene() : minp(Vector2f(0,0)), maxp(Vector2f(0,0)), ncircles(0) { }
        ~Scene() { Cudamap_free(&cm); }

        // --------- Geometry Manipulation ---------
        void addSegment(Line l, float res=0.01) {
            extendBbox(l.p1);
            extendBbox(l.p2);
            Vector2f v = l.vec();
            float d = l.length();
            Vector2f n = l.normal();
            for (float i = res/2; i < d; i+= res) {
                Vector2f p = l.p1 + i*v;
                surfels.push_back(p[0]);
                surfels.push_back(p[1]);
                surfels.push_back(n[0]);
                surfels.push_back(n[1]);
            }
            lines.push_back(l);
        }
        void addCircle(Vector2f o, float r, float res=0.01, bool flip=false) {
            circles.push_back(Vector3f(o[0], o[1], flip?-r:r));
            if (r < 0) {
                flip = !flip;
                r = -r;
            }
            extendBbox(o+Vector2f(r,r));
            extendBbox(o-Vector2f(r,r));
            float ares = asin(res/r);
            for (float a = 0; a < 2*M_PI; a += ares) {
                Vector2f n = sin(a)*Vector2f(0,1) + cos(a)*Vector2f(1,0);
                surfels.push_back(o[0] + n[0]*r);
                surfels.push_back(o[1] + n[1]*r);
                surfels.push_back(flip?n[0]:-n[0]);
                surfels.push_back(flip?n[1]:-n[1]);
            }
            ncircles++;
        }
        int numSurfels() const {
            return surfels.size()/4;
        }

        // --------- CUDA Setup ---------
        void initCuda(int w, int h) {
            cm.w = w;
            cm.h = h;
            field = new float[2*w*h];
            cm.maxx = maxp[0];
            cm.maxy = maxp[1];
            cm.minx = minp[0];
            cm.miny = minp[1];
            cm.n = surfels.size()/4;
            cm.nlines = lines.size();
            cm.ncircles = ncircles;
            vector<float> linedata;
            for (int i = 0; i < cm.nlines; i++) {
                linedata.push_back(lines[i].p1[0]);
                linedata.push_back(lines[i].p1[1]);
                linedata.push_back(lines[i].p2[0]);
                linedata.push_back(lines[i].p2[1]);
            }
            vector<float> circledata;
            for (int i = 0; i < cm.ncircles; i++) {
                circledata.push_back(circles[i][0]);
                circledata.push_back(circles[i][1]);
                circledata.push_back(abs(circles[i][2]));
                circledata.push_back(circles[i][2]);
            }
            Cudamap_init(&cm, surfels.data(), linedata.data(), circledata.data());
        }
        void setCudaGLTexture(GLuint tex) {
            Cudamap_setGLTexture(&cm, tex);
        }
        void setCudaGLBuffer(GLuint pbo) {
            Cudamap_setGLBuffer(&cm, pbo);
        }
        void computeField(float* distancefield=NULL) {
            Cudamap_computeField(&cm, distancefield?distancefield:field);
        }
        void computeDensity(float* density=NULL) {
            Cudamap_computeDensity(&cm, density?density:field);
        }

        // --------- Light Manipulation ---------
        int numLights() const {
            return lights.size();
        }
        Vector3f getLight(int idx) const {
            return lights[idx];
        }
        float getLightAngle(int idx) const {
            return directions[idx];
        }
        void addLightWithSymmetry(float intensity, int i) {
            if (directions[i] < 0) {
                Cudamap_addLight(&cm, intensity, lights[i][0], lights[i][1]);
                if (symmetries[i] & 1) {
                    Cudamap_addLight(&cm, intensity, -lights[i][0], lights[i][1]);
                }
                if (symmetries[i] & 2) {
                    Cudamap_addLight(&cm, intensity, lights[i][0], -lights[i][1]);
                }
                if (symmetries[i] & 4) {
                    Cudamap_addLight(&cm, intensity, -lights[i][0], -lights[i][1]);
                }
            } else {
                Cudamap_addDirectionalLight(
                        &cm, intensity, lights[i][0], lights[i][1],
                        cos(directions[i])*falloffs[i],
                        sin(directions[i])*falloffs[i]
                        );
            }
        }
        void changeIntensity(int i, float intensity) {
            addLightWithSymmetry(intensity-lights[i][2], i);
            lights[i][2] = intensity;
            //computeField();
        }
        void moveLight(float x, float y, int i) {
            addLightWithSymmetry(-lights[i][2], i);
            lights[i][0] = x;
            lights[i][1] = y;
            addLightWithSymmetry(lights[i][2], i);
            //computeField();
        }
        void changeDirection(float a, float falloff, int i) {
            if (directions[i] < 0) return;
            Cudamap_addDirectionalLight(
                    &cm, -lights[i][2], lights[i][0], lights[i][1],
                    cos(directions[i])*falloffs[i],
                    sin(directions[i])*falloffs[i]
                    );
            directions[i] = a;
            if (falloff > 0) falloffs[i] = falloff;
            Cudamap_addDirectionalLight(
                    &cm, lights[i][2], lights[i][0], lights[i][1],
                    cos(directions[i])*falloffs[i],
                    sin(directions[i])*falloffs[i]
                    );
            //computeField();
        }
        void deleteLight(int i) {
            addLightWithSymmetry(-lights[i][2], i);
            lights.erase(lights.begin()+i);
            directions.erase(directions.begin()+i);
            falloffs.erase(falloffs.begin()+i);
            symmetries.erase(symmetries.begin()+i);
            //computeField();
        }
        void addSymmetry(int i, int symm) {
            if (directions[i] >= 0) return;
            int newsym = symm - (symm & symmetries[i]);
            int oldsym = symm & symmetries[i];
            if (newsym & 1) {
                Cudamap_addLight(&cm, lights[i][2], -lights[i][0], lights[i][1]);
            }
            if (newsym & 2) {
                Cudamap_addLight(&cm, lights[i][2], lights[i][0], -lights[i][1]);
            }
            if (newsym & 4) {
                Cudamap_addLight(&cm, lights[i][2], -lights[i][0], -lights[i][1]);
            }
            if (oldsym & 1) {
                Cudamap_addLight(&cm, -lights[i][2], -lights[i][0], lights[i][1]);
            }
            if (oldsym & 2) {
                Cudamap_addLight(&cm, -lights[i][2], lights[i][0], -lights[i][1]);
            }
            if (oldsym & 4) {
                Cudamap_addLight(&cm, -lights[i][2], -lights[i][0], -lights[i][1]);
            }
            symmetries[i] |= newsym;
            symmetries[i] -= oldsym;
        }
        int getSymmetries(int i) const {
            return symmetries[i];
        }
        void saveLights() {
            savedLights = lights;
        }
        void restoreLights() {
            swap(lights, savedLights);
            float* intensities = new float[cm.n];
            memset(intensities, 0, sizeof(float)*cm.n);
            Cudamap_setIntensities(&cm, intensities);
            for (int i = 0; i < lights.size(); i++) {
                Cudamap_addLight(&cm, lights[i][2], lights[i][0], lights[i][1]);
            }
        }
        void addLight(float x, float y, float intensity=1) {
            Cudamap_addLight(&cm, intensity, x, y);
            lights.push_back(Vector3f(x,y,intensity));
            directions.push_back(-1);
            falloffs.push_back(-1);
            symmetries.push_back(0);
        }
        void addDirectionalLight(float x, float y, float a, float falloff, float intensity=1) {
            Cudamap_addDirectionalLight(&cm, intensity, x, y, cos(a)*falloff, sin(a)*falloff);
            lights.push_back(Vector3f(x,y,intensity));
            directions.push_back(a);
            falloffs.push_back(falloff);
            symmetries.push_back(0);
        }
        float computeError() {
            float total = 0;
            float n = 0;
            for (int i = 0; i < surfels.size(); i+=4) {
                float lighting = 0;
                float residual = 0;
                for (int j = 0; j < lights.size(); j++) {
                    // FIXME: non-point lights?
                    Vector2f L = lights[j].head(2) - Vector2f(surfels[i], surfels[i+1]);
                    float LdotL = L.squaredNorm();
                    float ndotL = Vector2f(surfels[i+2], surfels[i+3]).dot(L);
                    if (ndotL < 0) ndotL = 0;
                    float l = LdotL>0?ndotL*lights[j][2]/(LdotL*sqrt(LdotL)):0;
                    residual += l;
                    if (lights[j][2] > 0) lighting += l;
                }
                if (residual < 0) {
                    return -1;
                }
                total += lighting>0?abs(residual/lighting):0;
                n+=1;
            }
            return n>0?total/n:0;
        }

        // --------- Coordinate System Utilities ---------
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
        float sceneScale() const {
            Vector2f v = maxp-minp;
            return min(v[0], v[1]);
        }

        void printScene() {
            cout << circles.size() + lines.size() << " " << lights.size() << endl;
            for (int i = 0; i < lines.size(); i++) {
                //cout << 0 << " ";
                cout << lines[i].p1[0] << " " << lines[i].p1[1] << " ";
                cout << lines[i].p2[0] << " " << lines[i].p2[1] << endl;
            }
            for (int i = 0; i < circles.size(); i++) {
                //cout << 1;
                for (int j = 0; j < 3; j++) cout << " " << circles[i][j];
                cout << endl;
            }
            cout << endl;
            for (int i = 0; i < lights.size(); i++) {
                for (int j = 0; j < 3; j++) cout << lights[i][j] << " ";
                cout << endl;
            }
        }

    protected:
        void extendBbox(Vector2f p) {
            minp[0] = min(minp[0], p[0]);
            minp[1] = min(minp[1], p[1]);
            maxp[0] = max(maxp[0], p[0]);
            maxp[1] = max(maxp[1], p[1]);
        }

        Vector2f minp, maxp;

        vector<float> surfels;
        vector<Line> lines;
        vector<Vector3f> circles;
        int ncircles;

        vector<Vector3f> lights;
        vector<Vector3f> savedLights;
        vector<float> directions;
        vector<float> falloffs;
        vector<int> symmetries;

        Cudamap cm;
        float* field;
};

Scene s;
GLuint vao;
GLuint vbo[2];
GLuint pbo, tbo_tex, tex, auxtex;
GLuint rfr_tex, rfr_fbo_z, rfr_fbo;
int currprog;
enum {
    PROG_ID = 0,
    PROG_SOURCEMAP = 1,
    PROG_MEDIALAXIS = 2,
    PROG_DENSITY = 3,
    NUM_PROGS,
    PROG_GRAD = 4,
    PROG_LAPLACIAN = 5,
    PROG_LOCALMIN = 10,
    PROG_LOCALMAX = 11,
};
GLuint progs[NUM_PROGS];
float exposure;

int selectedlight = -1;
int dragging = 0;
Vector2f offset;

float* auxlayer;
const int RADIUS = 10;
const float ANGLEINC = M_PI/18.f;

void putpixel(float* arr, int w, int h, float v, int x, int y) {
    if (x < w && y < h && x >= 0 && y >= 0) arr[x+w*y] = v;
}
void rasterizeCircle(float* arr, int w, int h, int x0, int y0, int r, float v=0.4f) {
    int x = r;
    int y = 0;
    int d = 1-r;

    while (x >= y)
    {
        putpixel(arr, w, h, v, x0 + x, y0 + y);
        putpixel(arr, w, h, v, x0 + y, y0 + x);
        putpixel(arr, w, h, v, x0 - y, y0 + x);
        putpixel(arr, w, h, v, x0 - x, y0 + y);
        putpixel(arr, w, h, v, x0 - x, y0 - y);
        putpixel(arr, w, h, v, x0 - y, y0 - x);
        putpixel(arr, w, h, v, x0 + y, y0 - x);
        putpixel(arr, w, h, v, x0 + x, y0 - y);

        y++;
        if (d > 0) {
            x--;
            d += 2*(y-x)+1;
        } else {
            d += 2*y+1;
        }
    }
}

void rerasterizeLights() {
    memset(auxlayer, 0, width*height*displayscale*displayscale*sizeof(float));
    int ix, iy;
    int w = width*displayscale;
    int h = height*displayscale;
    for (int i = 0; i < s.numLights(); i++) {
        Vector2f p = s.getLight(i).head(2);
        s.world2clip(p, ix, iy, w, h);
        float a = s.getLightAngle(i);
        float color = 0.1f;
        if (selectedlight == i) color = 0.2f;
        else if (s.getLight(i)[2] < 0) {
            if (i == lastCandidate) color = 0.3f;
            else color = 0.4f;
        }
        rasterizeCircle(auxlayer, w, h, ix, iy, RADIUS, color);
        if (a >= 0) {
            rasterizeCircle(auxlayer, w, h, ix + RADIUS*cos(a), iy + RADIUS*sin(a), RADIUS/2, color);
        }

        if (s.getSymmetries(i) & 1) {
            s.world2clip(Vector2f(-p[0], p[1]), ix, iy, w, h);
            rasterizeCircle(auxlayer, w, h, ix, iy, RADIUS, color);
        }
        if (s.getSymmetries(i) & 2) {
            s.world2clip(Vector2f(p[0], -p[1]), ix, iy, w, h);
            rasterizeCircle(auxlayer, w, h, ix, iy, RADIUS, color);
        }
        if (s.getSymmetries(i) & 4) {
            s.world2clip(Vector2f(-p[0], -p[1]), ix, iy, w, h);
            rasterizeCircle(auxlayer, w, h, ix, iy, RADIUS, color);
        }
    }
    glBindTexture(GL_TEXTURE_2D, auxtex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, auxlayer);
    glBindTexture(GL_TEXTURE_2D, 0);
}
void selectLight(int i) {
    int ix, iy;
    if (selectedlight >= 0) {
        s.world2clip(s.getLight(selectedlight).head(2), ix, iy, width*displayscale, height*displayscale);
        rasterizeCircle(auxlayer, width*displayscale, height*displayscale, ix, iy, RADIUS, 0.4f);
    }
    s.world2clip(s.getLight(i<0?selectedlight:i).head(2), ix, iy, width*displayscale, height*displayscale);
    rasterizeCircle(auxlayer, width*displayscale, height*displayscale, ix, iy, RADIUS, i<0?0.4f:1.f);
    glBindTexture(GL_TEXTURE_2D, auxtex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width*displayscale, height*displayscale, GL_RED, GL_FLOAT, auxlayer);
    glBindTexture(GL_TEXTURE_2D, 0);
    selectedlight = i;
}

bool shouldExitImmediately = false;
bool shouldPrintSuccess = false;
bool shouldWriteExrFile = false;
bool shouldWritePngFile = false;
bool shouldWritePlyFile = false;
bool stepping = false;
vector<int> candidateLights;
string pngFilename, plyFilename, exrFilename;

void keydown(unsigned char key, int x, int y) {
    if (key == ',') {
        if (exposure > 0.05) exposure -= 0.05;
    } else if (key == '.') {
        exposure += 0.05;
    } else if (key == 'm') {
        currprog = (currprog+1)%NUM_PROGS;
    } else if (key == ' ') {
        if (pngFilename.length()) shouldWritePngFile = true;
        if (exrFilename.length()) shouldWriteExrFile = true;
        if (plyFilename.length()) shouldWritePlyFile = true;
    } else if (key == '[' && selectedlight >= 0) {
        float intensity = s.getLight(selectedlight)[2];
        if (intensity > 0.1) s.changeIntensity(selectedlight, intensity-0.1);
    } else if (key == ']' && selectedlight >= 0) {
        float intensity = s.getLight(selectedlight)[2];
        s.changeIntensity(selectedlight, intensity+0.1);
    } else if (key == 'q' && selectedlight >= 0) {
        float a = s.getLightAngle(selectedlight);
        if (a < 0) {
            s.addSymmetry(selectedlight, 1);
        } else {
            a -= ANGLEINC;
            if (a < 0) a += 2*M_PI;
            s.changeDirection(a, -1, selectedlight);
        }
        rerasterizeLights();
    } else if (key == 'w' && selectedlight >= 0) {
        float a = s.getLightAngle(selectedlight);
        if (a < 0) {
            s.addSymmetry(selectedlight, 2);
        } else {
            a += ANGLEINC;
            if (a > 2*M_PI) a -= 2*M_PI;
            s.changeDirection(a, -1, selectedlight);
        }
        rerasterizeLights();
    } else if (key == 'e' && selectedlight >= 0) {
        float a = s.getLightAngle(selectedlight);
        if (a < 0) {
            s.addSymmetry(selectedlight, 4);
            rerasterizeLights();
        }
    } else if (key == 'h') {
        cout << helpstring << endl;
    } else if (key == 'p') {
        s.printScene();
    } else if (key == 13) {
        stepping = !stepping;
    } else if (key == 127 && selectedlight >= 0) {
        s.deleteLight(selectedlight);
        rerasterizeLights();
        dragging = 0;
    } else if (key == 27) {
        exit(0);
    }
}

void click(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        Vector2f p = s.clip2world(x,height*displayscale-y,width*displayscale,height*displayscale);
        Vector2f p2 = s.clip2world(x+RADIUS,height*displayscale-y,width*displayscale,height*displayscale);
        float r = p2[0] - p[0];
        bool clicked = false;
        for (int i = 0; i < s.numLights(); i++) {
            if (s.getLight(i)[2] < 0) continue;
            if ((p-s.getLight(i).head(2)).squaredNorm() < r*r) {
                selectLight(i);
                clicked = true;
                break;
            }
        }
        if (!clicked) {
            if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
                s.addDirectionalLight(p[0], p[1], 0, 1);
            } else {
                s.addLight(p[0], p[1]);
            }
            rerasterizeLights();
            selectLight(s.numLights()-1);
        }
        dragging = 1;
        offset = s.getLight(selectedlight).head(2) - p;
    } else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
        dragging = 0;
    }
}
void mousemove(int x, int y) {
    if (dragging && selectedlight >= 0) {
        Vector2f p = s.clip2world(x,height*displayscale-y,width*displayscale,height*displayscale);
        p += offset;
        s.moveLight(p[0], p[1], selectedlight);
        rerasterizeLights();
    }
}

bool any(unsigned char* a, int w, int h, int x, int y, int d = 1) {
    for (int r = max(0,y-d); r < h && r <= y+d; r++) {
        for (int c = max(0,x-d); c < w && c <= x+d; c++) {
            if (a[3*(w*r+c)]) return true;
            //if (a[3*(w*(h-r-1)+c)]) return true;
        }
    }
    return false;
}

void GaussianBlur(float* a, float* b, int w, int h, int r, int nch=3) {
    float* G = new float[r+1];
    float tot = 0;
    for (int i = 0; i < r+1; i++) {
        float sigma = r/3.f;
        G[i] = exp(-i*i/(2*r*r));
        tot += 2*G[i];
    }
    tot -= 1;
    for (int i = 0; i < r+1; i++) G[i] /= tot;
    float* c = new float[nch*w*h];
    memset(c, 0, sizeof(float)*nch*w*h);
    // Horizontal
    for (int i = r; i < h-r; i++) {
        for (int j = r; j < w-r; j++) {
            for (int k = -r; k <= r; k++) {
                for (int ch = 0; ch < nch; ch++) c[nch*(i*w+j)+ch] += G[abs(k)]*a[nch*(i*w+j+k)+ch];
            }
        }
    }
    // Vertical
    for (int i = r; i < h-r; i++) {
        for (int j = r; j < w-r; j++) {
            for (int k = -r; k <= r; k++) {
                for (int ch = 0; ch < nch; ch++) b[nch*(i*w+j)+ch] += G[abs(k)]*c[nch*((i+k)*w+j)+ch];
            }
        }
    }
    delete [] c;
    delete [] G;
}

void recomputeMaxima(vector<Vector3f>& maxima) {
    maxima.clear();
    int ww = width;
    int hh = height;

    s.computeField(distancefield);
    s.computeDensity(imagecopy);
    GaussianBlur(imagecopy, filtered, ww, hh, 5,1);

    float lowthreshold = 0.5;
    int nbrhd = 15;
    int margin = 8;
    for (int r = margin; r < hh-margin; r++) {
        for (int c = margin; c < ww-margin; c++) {
            float fv = filtered[r*ww+c];
            if (fv > lowthreshold) {
                bool islocalmax = true;
                for (int i = -nbrhd; i <= nbrhd; i++) {
                    for (int j = -nbrhd; j <= nbrhd; j++) {
                        if (i == 0 && j == 0) continue;
                        float v = filtered[(r+i)*ww+c+j];
                        if (v >= fv) {
                            islocalmax = false;
                            break;
                        }
                    }
                    if (!islocalmax) break;
                }
                if (islocalmax) {
                    Vector2f p = s.clip2world(c,r,ww,hh);
                    maxima.push_back(Vector3f(p[0], p[1], distancefield[2*(ww*r+c)]));
                }
            }
        }
    }
}

bool updateEstimates() {
    double distancethreshold = 0.05*s.sceneScale();
    double decrement = -0.03;
    vector<Vector3f> maxima;
    vector<int> candidate2maximum(candidateLights.size(), -1);

    recomputeMaxima(maxima);

    // Associate maxima with existing lights, adding new lights if necessary
    for (int i = 0; i < maxima.size(); i++) {
        float closest = distancethreshold;
        int best = -1;
        Vector2f p = maxima[i].head(2);
        for (int j = 0; j < candidateLights.size(); j++) {
            Vector2f p2 = s.getLight(candidateLights[j]).head(2);
            float d = (p - p2).norm();
            if (d < closest) {
                closest = d;
                best = j;
            }
        }
        if (best >= 0) {
            if (candidate2maximum[best] < 0) {
                candidate2maximum[best] = i;
            } else {
                cout << "Error: lights too close together!" << endl;
            }
        } else {
            candidateLights.push_back(s.numLights());
            candidate2maximum.push_back(i);
            s.addLight(p[0], p[1], -EPSILON);
        }
    }

    // Decrease intensity of all detected maxima
    for (int i = 0; i < candidateLights.size(); i++) {
        if (candidate2maximum[i] < 0) continue;
        s.changeIntensity(candidateLights[i], s.getLight(candidateLights[i])[2] + decrement);
    }

    // Update positions of all lights
    vector<Vector2f> updatedPositions(candidateLights.size());
    for (int i = 0; i < candidateLights.size(); i++) {
        updatedPositions[i] = s.getLight(candidateLights[i]).head(2);
    }
    bool converged = false;
    float minLightMotion = 0.00002f;
    while (!converged) {
        float maxdelta = 0;
        vector<Vector3f> tmpmaxima;
        for (int i = 0; i < candidateLights.size(); i++) {
            // Reset the predicted intensity for this light
            // to compute better position with other light
            // intensities decreased
            float previousIntensity = s.getLight(candidateLights[i])[2];
            s.changeIntensity(candidateLights[i], -EPSILON);

            // Get updated position
            recomputeMaxima(tmpmaxima);
            float closest = distancethreshold;
            int best = -1;
            Vector2f p2 = s.getLight(candidateLights[i]).head(2);
            for (int j = 0; j < tmpmaxima.size(); j++) {
                float d = (tmpmaxima[j].head(2) - p2).squaredNorm();
                if (d < closest) {
                    closest = d;
                    best = j;
                }
            }
            if (best >= 0) {
                float d = (updatedPositions[i] - tmpmaxima[best].head(2)).squaredNorm();
                maxdelta = max(d, maxdelta);
                updatedPositions[i] = tmpmaxima[best].head(2);
            }

            // Reset intensities
            s.changeIntensity(candidateLights[i], previousIntensity);
        }
        if (maxdelta < minLightMotion) converged = true;
    }
    for (int i = 0; i < updatedPositions.size(); i++) {
        s.moveLight(updatedPositions[i][0], updatedPositions[i][1], candidateLights[i]);
        Vector3f l = s.getLight(candidateLights[i]);
        cout << i << " " << l[0] << " " << l[1] << " " << l[2];
        if (candidate2maximum[i] >= 0) cout << "(" << maxima[candidate2maximum[i]][2] << ")";
        cout << endl;
    }
    cout << "------------" << endl;
    return true;
}

void draw() {
    if (shouldWritePlyFile || shouldWriteExrFile) {
        s.computeField(distancefield);
        if (shouldWritePlyFile) {
            outputPLY(plyFilename.c_str(), distancefield, width, height, 2, displayscale==1?auxlayer:NULL);
            shouldWritePlyFile = false;
        }
        if (shouldWriteExrFile) {
            outputEXR(exrFilename.c_str(), distancefield, width, height, 2);
            shouldWriteExrFile = false;
        }
    } else {
        s.computeField();
    }
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    //glBindTexture(GL_TEXTURE_BUFFER,tbo_tex);
    //glTexBuffer(GL_TEXTURE_BUFFER,GL_R32F,pbo);
    //
    if (currprog == PROG_SOURCEMAP || currprog == PROG_MEDIALAXIS || currprog == PROG_DENSITY) {
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    } else {
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    if (currprog == PROG_LOCALMIN || currprog == PROG_LOCALMAX || currprog == PROG_LAPLACIAN) {
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, rfr_fbo);
        glUseProgram(progs[PROG_DENSITY]);
        glUniform1f(glGetUniformLocation(progs[PROG_DENSITY], "exposure"), 1);
        glUniform1i(glGetUniformLocation(progs[PROG_DENSITY], "maxidx"), s.numSurfels());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
        glDrawArrays(GL_TRIANGLES,0,6);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, rfr_tex);
    } else {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
    }
    glUseProgram(progs[currprog]);
    if (currprog == PROG_SOURCEMAP || currprog == PROG_MEDIALAXIS || currprog == PROG_DENSITY) {
        glUniform1i(glGetUniformLocation(progs[currprog], "maxidx"), s.numSurfels());
    }
    glUniform1f(glGetUniformLocation(progs[currprog], "exposure"), exposure);
    glActiveTexture(GL_TEXTURE1);
    if (shouldPrintSuccess) glBindTexture(GL_TEXTURE_2D, 0);
    else glBindTexture(GL_TEXTURE_2D, auxtex);
    glDrawArrays(GL_TRIANGLES,0,6);
    //glBindTexture(GL_TEXTURE_BUFFER,0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    if (shouldWritePngFile) {
        glReadPixels(0,0,width*displayscale, height*displayscale, GL_RGB, GL_UNSIGNED_BYTE, (void*) imagedata);
        outputPNG(pngFilename.c_str(), imagedata, width*displayscale, height*displayscale);
        shouldWritePngFile = false;
    }
    if (shouldExitImmediately) {
        exit(0);
    }
    if (stepping) {
        int ww = width;
        int hh = height;
        memset(filtered, 0, 3*ww*hh*sizeof(float));
        memset(imagecopy, 0, 3*ww*hh*sizeof(float));

        s.saveLights();
        if (!updateEstimates()) {
            cout << "Terminating: duplicated light" << endl;
            stepping = false;
        }
        float err = s.computeError();
        if (err < 0.05) {
            cout << "Terminating: negative incident illumination" << endl;
            stepping = false;
        }
        /*if (err > prevErr) {
            cout << "Terminating: rise in error " << prevErr << " to " << err <<  endl;
            stepping = false;
        }*/
        cout << err << endl;
        if (!stepping) {
            s.restoreLights();
            err = prevErr;
        } else {
            prevErr = err;
        }
        rerasterizeLights();
    }
    glutSwapBuffers();
}

void setupWindow(int argc, char** argv, int w, int h) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(w, h);
    glutCreateWindow("Light Localization");
    glutDisplayFunc(draw);
    glutIdleFunc(draw);
    glutKeyboardFunc(keydown);
    glutMouseFunc(click);
    glutMotionFunc(mousemove);
    openglInit();
}

void initRenderTextures() {
    glGenTextures(1, &rfr_tex);
    glBindTexture(GL_TEXTURE_2D, rfr_tex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width*displayscale, height*displayscale, 0, GL_RGBA, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenRenderbuffers(1, &rfr_fbo_z);
    glBindRenderbuffer(GL_RENDERBUFFER, rfr_fbo_z);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width*displayscale, height*displayscale);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glGenFramebuffers(1, &rfr_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, rfr_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rfr_tex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rfr_fbo_z);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void initCudaGlTextures() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 2*width*height*sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glGenTextures(1, &tbo_tex);
    glBindTexture(GL_TEXTURE_BUFFER, tbo_tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, pbo);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void setupProg(const char* fshader, int n) {
    ShaderProgram* prog;
    prog = new FileShaderProgram("tboshader.v.glsl", fshader);
    prog->init();
    progs[n] = prog->getProgId();
    delete prog;
    glUseProgram(progs[n]);
    glUniform1i(glGetUniformLocation(progs[n], "buffer"), 0);
    glUniform1i(glGetUniformLocation(progs[n], "aux"), 1);
    glUniform2i(glGetUniformLocation(progs[n], "dim"), width, height);
    glUniform1f(glGetUniformLocation(progs[n], "exposure"), exposure);
    glUniform1i(glGetUniformLocation(progs[n], "threshold"), 10);
}

void setupFullscreenQuad() {
    float points[] =  {
        -1.f, -1.f, 0.f,
        1.f, -1.f, 0.f,
        1.f, 1.f, 0.f,
        -1.f, -1.f, 0.f,
        1.f, 1.f, 0.f,
        -1.f, 1.f, 0.f
    };
    float texcoords[] = {
        0.f, 0.f,
        1.f, 0.f,
        1.f, 1.f,
        0.f, 0.f,
        1.f, 1.f,
        0.f, 1.f
    };
    glGenVertexArrays(1, &vao);
    glGenBuffers(2, vbo);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(float), points, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, 6*2*sizeof(float), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

bool endswith(const string& s, string e) {
    if (s.length() > e.length())
        return s.compare(s.length()-e.length(), e.length(), e) == 0;
    else
        return false;
}

int main(int argc, char** argv) {
    option::Stats stats(usage, argc-1, argv+1);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(usage, argc-1, argv+1, options, buffer);
    if (parse.error()) {
        option::printUsage(cout, usage);
        return 1;
    }
    if (options[RESOLUTION]) {
        width = atoi(options[RESOLUTION].arg);
        height = width;
        displayscale = 900/width;
    }
    if (options[DISPLAYSCALE]) {
        displayscale = atoi(options[DISPLAYSCALE].arg);
    }

    int ww = width*displayscale;
    int hh = height*displayscale;
    setupWindow(argc, argv, ww, hh);
    imagedata = new unsigned char[3*ww*hh];
    medialaxis = new unsigned char[3*ww*hh];
    filtered = new float[3*ww*hh];
    imagecopy = new float[3*ww*hh];
    distancefield = new float[2*width*height];
    auxlayer = new float[ww*hh];
    memset(auxlayer, 0, ww*hh*sizeof(float));
    glGenTextures(1, &auxtex);
    glBindTexture(GL_TEXTURE_2D, auxtex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, ww, hh, 0, GL_RED, GL_FLOAT, auxlayer);
    glBindTexture(GL_TEXTURE_2D, 0);

    initCudaGlTextures();
    initRenderTextures();
    setupFullscreenQuad();

    exposure = 0.5;

    setupProg("tboshader.f.glsl",PROG_ID);
    //setupProg("grad.f.glsl",PROG_GRAD);
    //setupProg("grad.f.glsl",PROG_LAPLACIAN);
    //setupProg("localmin.f.glsl",PROG_LOCALMIN);
    setupProg("sourcemap.f.glsl",PROG_SOURCEMAP);
    setupProg("medialaxis.f.glsl",PROG_MEDIALAXIS);
    setupProg("density.f.glsl",PROG_DENSITY);
    currprog = 0;
    if (options[MODE]) {
        currprog = atoi(options[MODE].arg)%NUM_PROGS;
    }

    if (options[INPUT_SCENEFILE]) {
        ifstream in(options[INPUT_SCENEFILE].arg);
        int nsegs, nlights, type;
        float x, y, z;
        in >> nsegs >> nlights;
        for (int i = 0; i < nsegs; i++) {
            type = 0;
            //in >> type;
            if (type == 0) {
                in >> x >> y;
                Vector2f v1(x,y);
                in >> x >> y;
                Vector2f v2(x,y);
                s.addSegment(Line(v1,v2));
            } else {
                in >> x >> y >> z;
                s.addCircle(Vector2f(x,y),z);
            }
        }
        s.initCuda(width, height);
        s.setCudaGLTexture(tex);
        s.setCudaGLBuffer(pbo);
        for (int i = 0; i < nlights; i++) {
            in >> x >> y >> z;
            s.addLight(x, y, z);
        }
    } else {
        s.addCircle(Vector2f(0,0), 1.0f, 0.007f);
        //s.addSegment(Line(Vector2f(-1, -1.01), Vector2f(-1, 1.01)));
        //s.addSegment(Line(Vector2f(-1.01, 1), Vector2f(1.01, 1)));
        //s.addSegment(Line(Vector2f(1, 1.01), Vector2f(1, -1.01)));
        //s.addSegment(Line(Vector2f(1.01, -1), Vector2f(-1.01, -1)));
        s.initCuda(width, height);
        s.setCudaGLTexture(tex);
        s.setCudaGLBuffer(pbo);

        s.addLight(0,0);
    }

    if (options[EXIT_IMMEDIATELY]) {
        shouldExitImmediately = true;
    }
    if (options[PRINT_SUCCESS]) {
        shouldPrintSuccess = true;
        currprog = PROG_DENSITY;
    }
    if (options[OUTPUT_IMAGEFILE]) {
        string s = options[OUTPUT_IMAGEFILE].arg;
        if (endswith(s, ".exr")) {
            exrFilename = s;
            if (shouldExitImmediately) shouldWriteExrFile = true;
        } else if (endswith(s, ".png")) {
            pngFilename = s;
            if (shouldExitImmediately) shouldWritePngFile = true;
        } else {
            cout << "Unknown image output format" << endl;
        }
    }
    if (options[OUTPUT_MESHFILE]) {
        plyFilename = options[OUTPUT_MESHFILE].arg;
        if (shouldExitImmediately) shouldWritePlyFile = true;
    }

    rerasterizeLights();

    glutMainLoop();
}
