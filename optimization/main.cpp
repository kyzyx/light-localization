#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/Eigen>

#include "geometry.h"
#include "solveCeres.h"
#include <ceres/ceres.h>
using namespace std;
using namespace Eigen;

default_random_engine generator;
uniform_real_distribution<double> dist(-1,1);
double randf() {
    return dist(generator);
}

class Scene {
    public:
        Scene() : minp(Vector2f(0,0)), maxp(Vector2f(0,0)), ncircles(0) { }
        ~Scene() {}

        // --------- Geometry Manipulation ---------
        void addSegment(Line l, double res=0.01) {
            extendBbox(l.p1);
            extendBbox(l.p2);
            Vector2f v = l.vec();
            double d = l.length();
            Vector2f n = l.normal();
            for (double i = res/2; i < d; i+= res) {
                Vector2f p = l.p1 + i*v;
                surfels.push_back(p[0]);
                surfels.push_back(p[1]);
                surfels.push_back(n[0]);
                surfels.push_back(n[1]);
            }
            lines.push_back(l);
        }
        void addCircle(Vector2f o, double r, double res=0.01, bool flip=false) {
            circles.push_back(Vector3f(o[0], o[1], flip?-r:r));
            if (r < 0) {
                flip = !flip;
                r = -r;
            }
            extendBbox(o+Vector2f(r,r));
            extendBbox(o-Vector2f(r,r));
            double ares = asin(res/r);
            for (double a = 0; a < 2*M_PI; a += ares) {
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
        int numLights() const {
            return lights.size();
        }
        Vector3f getLight(int idx) const {
            return lights[idx];
        }
        double getLightAngle(int idx) const {
            return directions[idx];
        }
        void addLight(double x, double y, double intensity=1) {
            lights.push_back(Vector3f(x,y,intensity));
            directions.push_back(-1);
            falloffs.push_back(-1);
            symmetries.push_back(0);
        }
        void computeLighting() {
            intensities.clear();
            int dim = 2;
            for (int i = 0; i < surfels.size(); i += 2*dim) {
                double tot = 0;
                for (int j = 0; j < lights.size(); j++) {
                    double LdotL = 0;
                    double ndotLn = 0;
                    for (int k = 0; k < dim; k++) {
                        double L = lights[j][k]-surfels[i+k];
                        ndotLn += surfels[i+dim+k]*L;
                        LdotL += L*L;
                    }
                    ndotLn /= sqrt(LdotL);
                    tot += ndotLn>0?lights[j][dim]*ndotLn/LdotL:0;
                }
                intensities.push_back(tot);
            }
        }
        void extendBbox(Vector2f p) {
            minp[0] = min(minp[0], p[0]);
            minp[1] = min(minp[1], p[1]);
            maxp[0] = max(maxp[0], p[0]);
            maxp[1] = max(maxp[1], p[1]);
        }

        Vector2f minp, maxp;
        vector<double> surfels;
        vector<double> intensities;
        vector<Line> lines;
        vector<Vector3f> circles;
        int ncircles;

        vector<Vector3f> lights;
        vector<double> directions;
        vector<double> falloffs;
        vector<int> symmetries;
};

Scene s;

void readPly(const char* filename,
        std::vector<float>& verts,
        std::vector<float>& norms,
        std::vector<unsigned int>& faces,
        std::vector<float>& colors,
        std::vector<float>& lights
        )
{
    ifstream in(filename);
    int nfaces, nvertices;
    int nlights = 0;
    bool hasnormals = true;
    bool hascolors = false;
    // Parse Header
    string line;
    string s;
    getline(in, line);
    if (line == "ply") {
        getline(in, line);
        string element;
        while (line != "end_header") {
            stringstream sin(line);
            sin >> s;
            if (s == "element") {
                sin >> s;
                if (s == "vertex") {
                    element = "vertex";
                    sin >> nvertices;
                } else if (s == "face") {
                    element = "face";
                    sin >> nfaces;
                } else if (s == "light") {
                    element = "light";
                    sin >> nlights;
                }
            } else if (s == "property" && element == "vertex") {
                sin >> s; // Data type
                sin >> s; // name
                if (s == "red") {
                    hascolors = true;
                }
            }
            getline(in, line);
        }
    } else {
        cerr << "Error parsing PLY header" << endl;
    }
    for (int i = 0; i < nvertices; i++) {
        float x, y, z;
        in >> x >> y >> z;
        verts.push_back(x);
        verts.push_back(y);
        verts.push_back(z);
        if (hasnormals) {
            in >> x >> y >> z;
            norms.push_back(x);
            norms.push_back(y);
            norms.push_back(z);
        }
        if (hascolors) {
            in >> x >> y >> z;
            colors.push_back(x);
            colors.push_back(y);
            colors.push_back(z);
        }
    }
    for (int i = 0; i < nfaces; i++) {
        float a, b, c, d;
        in >> a >> b >> c >> d;
        faces.push_back(b);
        faces.push_back(c);
        faces.push_back(d);
    }
    for (int i = 0; i < nlights; i++) {
        float a, b, c, d;
        in >> a >> b >> c >> d;
        lights.push_back(a);
        lights.push_back(b);
        lights.push_back(c);
        lights.push_back(d);
    }
}

int main(int argc, char** argv) {
    double* lightparams;
    double* lightintensities;
    double* geometry;
    double* intensities;
    std::vector<float> verts;
    std::vector<float> norms;
    std::vector<unsigned int> faces;
    std::vector<float> colors;
    std::vector<float> lights;
    if (argc > 1) {
        readPly(argv[1], verts, norms, faces, colors, lights);
    } else {
        cout << "Usage: optimization filename.ply" << endl;
        exit(0);
    }
    int DIM = 3;
    int nl = lights.size()/(DIM+1);
    int nv = verts.size()/DIM;
    lightparams = new double[nl*DIM];
    lightintensities = new double[nl];
    geometry = new double[2*verts.size()];
    intensities = new double[nv];
    int z = 0;
    for (int i = 0; i < verts.size(); i+=DIM) {
        for (int j = 0; j < DIM; j++) geometry[z++] = verts[i+j];
        for (int j = 0; j < DIM; j++) geometry[z++] = norms[i+j];
    }
    if (colors.empty()) {
        colors.resize(verts.size(),0);
        for (int i = 0; i < verts.size(); i+=3) {
            for (int j = 0; j < lights.size(); j+=4) {
                float LdotL = 0;
                float ndotLn = 0;
                for (int k = 0; k < 3; k++) {
                    float L = lights[j+k]-verts[i+k];
                    ndotLn += norms[i+k]*L;
                    LdotL += L*L;
                }
                ndotLn /= sqrt(LdotL);
                float color = ndotLn>0?lights[j+3]*ndotLn/LdotL:0;
                for (int k = 0; k < 3; k++) colors[i+k] += color;
            }
        }
    }
    for (int i = 0; i < colors.size(); i+=DIM) {
        intensities[i/DIM] = colors[i];
    }
    generator.seed(time(0));
    int success = 0;
    int num_trials = 50;
    google::InitGoogleLogging("solveCeres()");
    for (int i = 0; i < num_trials; i++) {
        for (int j = 0; j < nl; j++) {
            for (int k = 0; k < DIM; k++) {
                lightparams[DIM*j+k] = randf();
            }
            lightintensities[j] = 1;
        }

        double costi = solveIntensitiesCeres(geometry, intensities, nv,
                lightparams, lightintensities, nl);
        double costj = solveCeres(geometry, intensities, nv,
                lightparams, lightintensities, nl);
        //cout << "Run " << i+1 << " " ;
        /*cout << costj << " " << costi << " { ";
        for (int j = 0; j < s.lights.size(); j++) {
            for (int k = 0; k < 2; k++) cout << lightparams[2*j+k] << " ";
            cout << lightintensities[j] << endl;
        }
        cout << " }" << endl;*/
        if (costj < 1) success++;
    }
    cout << "Real success rate: " <<  success / (float) num_trials << endl;
}
