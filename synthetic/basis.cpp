#include "R3Graphics/R3Graphics.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: basis input.ply spatial_resolution angular_bins > lights.txt" << endl;
        return 0;
    }
    R3Mesh mesh;
    mesh.ReadPlyFile(argv[1]);
    double spatialres = atof(argv[2]);
    int angularres = atof(argv[3]);
    int lighttype;
    if (angularres == 1) {
        lighttype = 0;
    } else {
        if (angularres != 12) {
            cerr << "Not supported - using 12 angular bins" << endl;
        }
        lighttype = 1;
    }

    R3Box bb = mesh.BBox();
    int xl = bb.XLength()/spatialres - 1;
    int yl = bb.YLength()/spatialres - 1;
    int zl = bb.ZLength()/spatialres - 1;
    double xs = bb.XMin() + (bb.XLength() - xl*spatialres)/2;
    double ys = bb.YMin() + (bb.YLength() - yl*spatialres)/2;
    double zs = bb.ZMin() + (bb.ZLength() - zl*spatialres)/2;
    cerr << (xl*yl*zl) << " points" << endl;

    cout << (xl*yl*zl*angularres) << endl;

    vector<R3Vector> dirs;
    for (int i = 0; i < 3; i++) {
        for (int s1 = -1; s1 <= 1; s1+=2) {
            for (int s2 = -1; s2 <= 1; s2+=2) {
                R3Vector v(0,0,0);
                v[i] = 1*s1;
                v[(i+1)%3] = 1.618034*s2;
                v.Normalize();
                dirs.push_back(v);
            }
        }
    }

    for (int a = 0; a < xl; a++) {
        for (int b = 0; b < yl; b++) {
            for (int c = 0; c < zl; c++) {
                if (lighttype == 1) {
                    for (int d = 0; d < angularres; d++) {
                        // Type and intensity
                        cout << lighttype << " 1 ";
                        // Position
                        cout << xs + a*spatialres << " ";
                        cout << ys + b*spatialres << " ";
                        cout << zs + c*spatialres << " ";
                        // Cutoff and exponent
                        cout << "\t1.4 1\t";
                        // Direction
                        cout << dirs[d][0] << " ";
                        cout << dirs[d][1] << " ";
                        cout << dirs[d][2] << " ";
                        cout << endl;
                    }
                } else {
                    // Type and intensity
                    cout << lighttype << " 1 ";
                    // Position
                    cout << xs + a*spatialres << " ";
                    cout << ys + b*spatialres << " ";
                    cout << zs + c*spatialres << " ";
                    cout << endl;
                }
            }
        }
    }
    return 0;
}
