#include "mesh.h"
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "loadshader.h"

using namespace std;

LitMesh::LitMesh(Cudamap* cudamap)
    : cm(cudamap)
{
}

LitMesh::~LitMesh() {
}

void readPly(const char* filename,
        std::vector<float>& verts,
        std::vector<float>& norms,
        std::vector<unsigned int>& faces,
        std::vector<float>& lights
        )
{
    ifstream in(filename);
    int nfaces, nvertices;
    int nlights = 0;
    // Parse Header
    string line;
    string s;
    getline(in, line);
    if (line == "ply") {
        getline(in, line);
        while (line != "end_header") {
            stringstream sin(line);
            sin >> s;
            if (s == "element") {
                sin >> s;
                if (s == "vertex") {
                    sin >> nvertices;
                } else if (s == "face") {
                    sin >> nfaces;
                } else if (s == "light") {
                    sin >> nlights;
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
        in >> x >> y >> z;
        norms.push_back(x);
        norms.push_back(y);
        norms.push_back(z);
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

void initAO(GLuint* vao,
        int nv, const float* pos, const float* normal,
        int nf, const unsigned int* faces,
        const float* col=0)
{
    GLuint vbo[3];
    GLuint ibo;

    glGenVertexArrays(1, vao);
    glGenBuffers(3, vbo);
    glGenBuffers(1, &ibo);

    glBindVertexArray(vao[0]);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, nv*3*sizeof(float), pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, nv*3*sizeof(float), normal, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    if (col) {
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
        glBufferData(GL_ARRAY_BUFFER, nv*3*sizeof(float), col, GL_STATIC_DRAW);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, nf*sizeof(unsigned int), faces, GL_STATIC_DRAW);
    glBindVertexArray(0);
}

void LitMesh::ReadFromPly(const char* filename) {
    readPly(filename, v, n, f, l);
    computeLighting();
    initShaders();
    initAO(&meshao, v.size()/3, v.data(), n.data(), f.size(), f.data(), c.data());

    vector<float> sv, sn, sl;
    vector<unsigned int> sf;
    readPly("sphere.ply", sv, sn, sf, sl);
    initAO(&sphereao, sv.size()/3, sv.data(), sn.data(), sf.size(), sf.data());
    numspherefaces = sf.size();
}

void LitMesh::cudaInit(int w, int h) {
    cm->w = w;
    cm->h = h;
    cm->n = v.size()/3;
    Cudamap_init(cm, v.data(), n.data());
    for (int i = 0; i < l.size(); i += 4) {
        Cudamap_addLight(cm, l[i+3], l[i], l[i+1], l[i+2]);
    }
}

void LitMesh::initShaders() {
    ShaderProgram* meshprog = new FileShaderProgram("pass.v.glsl", "pass.f.glsl");
    meshprog->init();
    meshprogid = meshprog->getProgId();
    meshmvmatrixuniform = glGetUniformLocation(meshprogid, "modelviewmatrix");
    meshprojectionmatrixuniform = glGetUniformLocation(meshprogid, "projectionmatrix");
    delete meshprog;
    ShaderProgram* lightprog = new FileShaderProgram("pass.v.glsl", "fixedlight.f.glsl");
    lightprog->init();
    lightprogid = lightprog->getProgId();
    lightmvmatrixuniform = glGetUniformLocation(lightprogid, "modelviewmatrix");
    lightprojectionmatrixuniform = glGetUniformLocation(lightprogid, "projectionmatrix");
    lightposuniform = glGetUniformLocation(lightprogid, "translation");
    delete lightprog;
}


void LitMesh::computeLighting() {
    c.resize(v.size(),0);
    for (int i = 0; i < v.size(); i+=3) {
        for (int j = 0; j < l.size(); j+=4) {
            float LdotL = 0;
            float ndotLn = 0;
            for (int k = 0; k < 3; k++) {
                float L = l[j+k]-v[i+k];
                ndotLn += n[i+k]*L;
                LdotL += L*L;
            }
            ndotLn /= sqrt(LdotL);
            float color = ndotLn>0?l[j+3]*ndotLn/LdotL:0;
            for (int k = 0; k < 3; k++) c[i+k] += color;
        }
    }
}

void LitMesh::Render() {
    GLfloat modelview[16];
    GLfloat projection[16];

    glUseProgram(meshprogid);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
    glGetFloatv(GL_PROJECTION_MATRIX, projection);
    glUniformMatrix4fv(meshprojectionmatrixuniform, 1, GL_FALSE, projection);
    glUniformMatrix4fv(meshmvmatrixuniform, 1, GL_FALSE, modelview);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_LIGHTING);
    glBindVertexArray(meshao);
    glDrawElements(GL_TRIANGLES, f.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void LitMesh::RenderLights(float radius) {
    GLfloat modelview[16];
    GLfloat projection[16];

    glUseProgram(lightprogid);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glBindVertexArray(sphereao);
    glMatrixMode(GL_MODELVIEW);
    for (int i = 0; i < l.size(); i += 4) {
        glPushMatrix();
        glTranslatef(l[i], l[i+1], l[i+2]);
        glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
        glGetFloatv(GL_PROJECTION_MATRIX, projection);
        glUniformMatrix4fv(lightprojectionmatrixuniform, 1, GL_FALSE, projection);
        glUniformMatrix4fv(lightmvmatrixuniform, 1, GL_FALSE, modelview);
        glDrawElements(GL_TRIANGLES, numspherefaces, GL_UNSIGNED_INT, 0);
        glPopMatrix();
    }
}
