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
    gluDeleteQuadric(quadric);
}

void LitMesh::ReadFromPly(const char* filename) {
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
        v.push_back(x);
        v.push_back(y);
        v.push_back(z);
        in >> x >> y >> z;
        n.push_back(x);
        n.push_back(y);
        n.push_back(z);
    }
    for (int i = 0; i < nfaces; i++) {
        float a, b, c, d;
        in >> a >> b >> c >> d;
        f.push_back(b);
        f.push_back(c);
        f.push_back(d);
    }
    for (int i = 0; i < nlights; i++) {
        float a, b, c, d;
        in >> a >> b >> c >> d;
        addLight(d,a,b,c);
    }

    computeLighting();
    initOpenGL();
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
    //ShaderProgram* lightprog = new FileShaderProgram("pass.v.glsl", "fixedlight.f.glsl");
    ShaderProgram* lightprog = new FileShaderProgram("pass.v.glsl", "pass.f.glsl");
    lightprog->init();
    lightprogid = lightprog->getProgId();
    lightmvmatrixuniform = glGetUniformLocation(lightprogid, "modelviewmatrix");
    lightprojectionmatrixuniform = glGetUniformLocation(lightprogid, "projectionmatrix");
    delete lightprog;
}

void LitMesh::initOpenGL() {
    initShaders();

    quadric = gluNewQuadric();
    gluQuadricDrawStyle(quadric, GLU_FILL);

    glGenVertexArrays(1, &vao);
    glGenBuffers(3, vbo);
    glGenBuffers(1, &ibo);

    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, v.size()*sizeof(float), v.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, n.size()*sizeof(float), n.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, c.size()*sizeof(float), c.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, f.size()*sizeof(unsigned int), f.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

void LitMesh::computeLighting() {
    c.resize(v.size());
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
            for (int k = 0; k < 3; k++) c[i+k] = color;
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
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, f.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void LitMesh::RenderLights(float radius) {
    /*GLfloat modelview[16];
    GLfloat projection[16];

    glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
    glGetFloatv(GL_PROJECTION_MATRIX, projection);
    glUniformMatrix4fv(lightprojectionmatrixuniform, 1, GL_FALSE, projection);
    glUniformMatrix4fv(lightmvmatrixuniform, 1, GL_FALSE, modelview);*/

    glUseProgram(0);
    glPointSize(2.f);
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glBegin(GL_POINTS);
    for (int i = 0; i < l.size(); i += 4) {
        glColor3f(1.f,0.f,0.f);
        glVertex3f(l[i+0], l[i+1], l[i+2]);
    }
    glEnd();
    /*glUseProgram(lightprogid);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glMatrixMode(GL_MODELVIEW);
    for (int i = 0; i < l.size(); i += 4) {
        glPushMatrix();
        glTranslatef(-l[i+0], -l[i+1], -l[i+2]);
        gluSphere(quadric, radius, 20, 20);
        glPopMatrix();
    }*/
}
