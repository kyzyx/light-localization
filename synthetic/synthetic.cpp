#include "R3Graphics/R3Graphics.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class Light {
    public:
        Light()
            : intensity(1), position(0,0,0)
        {}
        Light(R3Point pos, double l)
            : intensity(l), position(pos)
        {}

        virtual double L(const R3Point& p, const R3Vector& v) const {
            R3Vector lv = position - p;
            double d2 = lv.Dot(lv);
            lv.Normalize();
            double cl = lv.Dot(v);
            if (cl < 0) return 0;
            return cl*intensity/d2;
        }

        virtual void read(istream& in) {
            in >> intensity;
            in >> position[0] >> position[1] >> position[2];
        }

        double intensity;
        R3Point position;
};

class SpotLight : public Light {
    public:
        SpotLight() {}

        virtual double L(const R3Point& p, const R3Vector& v) const {
            R3Vector lv = position - p;
            double d2 = lv.Dot(lv);
            lv.Normalize();
            double cl = lv.Dot(v);
            if (cl < 0) return 0;
            double cd = -lv.Dot(direction);
            double a = acos(cd);
            if (a > cutoff) return 0;
            return pow(fabs(cd),exponent)*cl*intensity/d2;
        }

        virtual void read(istream& in) {
            Light::read(in);
            in >> cutoff >> exponent;
            in >> direction[0] >> direction[1] >> direction[2];
            direction.Normalize();
        }
        double exponent;
        double cutoff;
        R3Vector direction;
};
class LineLight : public Light {
    public:
        LineLight() {}

        virtual double L(const R3Point& p, const R3Vector& v) const {
            R3Vector lv = position - p;
            double d2 = lv.Dot(lv);
            lv.Normalize();
            double cl = lv.Dot(v);
            if (cl < 0) return 0;
            double cd = direction.Dot(lv);
            return sqrt(1-cd*cd)*cl*intensity/d2;
        }

        virtual void read(istream& in) {
            Light::read(in);
            in >> direction[0] >> direction[1] >> direction[2];
            direction.Normalize();
        }
        R3Vector direction;
};

bool visible(R3Mesh& m, R3Point p1, R3Point p2) {
    return true;
}

void writePlyMesh(R3Mesh& m, const string& filename, vector<double> customcolors, double scalefactor=1, double gamma=1, bool hdr=false) {
    ofstream out(filename);
    out << "ply" << endl;
    out << "format ascii 1.0" << endl;
    out << "element vertex " << m.NVertices() << endl;
    out << "property float x" << endl;
    out << "property float y" << endl;
    out << "property float z" << endl;
    if (hdr) {
        out << "property float red" << endl;
        out << "property float green" << endl;
        out << "property float blue" << endl;
    } else {
        out << "property uchar red" << endl;
        out << "property uchar green" << endl;
        out << "property uchar blue" << endl;
    }
    out << "element face " << m.NFaces() << endl;
    out << "property list uchar int vertex_indices" << endl;
    out << "end_header" << endl;
    for (int i = 0; i < m.NVertices(); ++i) {
        R3Point p = m.VertexPosition(m.Vertex(i));
        out << p[0] << " " << p[1] << " " << p[2] << " ";
        double c = customcolors[i];
        if (hdr) {
            c = pow(c,gamma)*scalefactor;
        } else {
            c = pow(c,gamma)*255*scalefactor;
            c = min((int)c, 255);
        }
        out << c << " " << c << " " << c << endl;
    }
    for (int i = 0; i < m.NFaces(); ++i) {
        out << "3";
        for (int j = 0; j < 3; ++j) {
            int vid = m.VertexID(m.VertexOnFace(m.Face(i), j));
            out << " " << vid;
        }
        out << endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: synthetic input.ply lights.txt [output.ply] > output.m" << endl;
        return 0;
    }
    R3Mesh mesh;
    mesh.ReadPlyFile(argv[1]);
    ifstream in(argv[2]);
    int nlights;
    in >> nlights;
    vector<Light*> lights;
    for (int i = 0; i < nlights; i++) {
        int type;
        in >> type;
        Light* l;
        if (type == 0) l = new Light();
        else if (type == 1) l = new SpotLight();
        else if (type == 2) l = new LineLight();
        l->read(in);
        lights.push_back(l);
    }

    vector<double> colors(mesh.NVertices());
    for (int i = 0; i < mesh.NVertices(); i++) {
        double radiance = 0;
        for (int j = 0; j < lights.size(); j++) {
            R3MeshVertex* v = mesh.Vertex(i);
            if (visible(mesh, mesh.VertexPosition(v), lights[j]->position)) {
                radiance += lights[j]->L(mesh.VertexPosition(v), mesh.VertexNormal(v));
            }
        }
        colors[i] = radiance;
        cout << radiance << endl;
    }
    if (argc > 3) {
        writePlyMesh(mesh, argv[3], colors);
    }
    return 0;
}
