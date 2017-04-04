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
            for (double i = 0; i < d; i+= res) {
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


int main(int argc, char** argv) {
    double* lightparams;
    double* lightintensities;
    double* geometry;
    double* intensities;
    if (argc > 1) {
        ifstream in(argv[1]);
        int nsegs, nlights, type;
        double x, y, z;
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
        for (int i = 0; i < nlights; i++) {
            in >> x >> y >> z;
            s.addLight(x, y, z);
        }
        lightparams = new double[s.lights.size()*2];
        lightintensities = new double[s.lights.size()];
        s.computeLighting();
    } else {
        s.addCircle(Vector2f(0,0), 1.0f, 0.007f);
        s.addLight(0,0);
        s.computeLighting();
        lightparams = new double[s.lights.size()*2];
        lightintensities = new double[s.lights.size()];
    }
    geometry = new double[s.surfels.size()];
    intensities = new double[s.intensities.size()];
    generator.seed(time(0));
    int success = 0;
    int num_trials = 500;
    google::InitGoogleLogging("solveCeres()");
    for (int i = 0; i < num_trials; i++) {
        memcpy(geometry, s.surfels.data(), sizeof(double)*s.surfels.size());
        memcpy(intensities, s.intensities.data(), sizeof(double)*s.intensities.size());
        double costi = solveIntensitiesCeres(geometry, intensities, s.intensities.size(),
                lightparams, lightintensities, s.lights.size());
        double costj = solveCeres(geometry, intensities, s.intensities.size(),
                lightparams, lightintensities, s.lights.size());
        //cout << "Run " << i+1 << " " ;
        /*cout << costj << " " << costi << " { ";
        for (int j = 0; j < s.lights.size(); j++) {
            for (int k = 0; k < 2; k++) cout << lightparams[2*j+k] << " ";
            cout << lightintensities[j] << endl;
        }
        cout << " }" << endl;*/
        if (costj < 1) success++;
    }
    cout << success / (float) num_trials << endl;
}
