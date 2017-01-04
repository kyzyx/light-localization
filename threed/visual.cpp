#include "opengl_compat.h"
#include <iostream>
#include <vector>
#include <fstream>

#include "planemanager.h"
#include "loadshader.h"
#include "cudamap.h"
#include "options.h"
#include "fileio.h"
#include "mesh.h"
#include "trackball.h"

int width = 600;
int height = 600;
int width3d = 600;
int height3d = 600;
unsigned char* imagedata;
float* distancefield;

using namespace std;

PlaneManager* planemanager;
Cudamap cudamap;
LitMesh* mesh;

GLuint vao;
GLuint vbo[2];
GLuint pbo, tbo_tex, tex;
GLuint rfr_tex, rfr_fbo_z, rfr_fbo;
int currprog;
enum {
    PROG_ID = 0,
    PROG_SOURCEMAP = 1,
    PROG_MEDIALAXIS = 2,
    NUM_PROGS,
};
GLuint progs[NUM_PROGS];
float exposure;

bool shouldExitImmediately = false;
bool shouldWriteExrFile = false;
bool shouldWritePngFile = false;
bool shouldWritePlyFile = false;
string pngFilename, plyFilename, exrFilename;

void keydown(unsigned char key, int x, int y) {
    if (key == ',') {
        if (exposure > 0.5) {
            exposure -= 0.5;
            planemanager->setExposure(exposure);
        }
    } else if (key == '.') {
        exposure += 0.5;
        planemanager->setExposure(exposure+0.5);
    } else if (key == '=') {
        mesh->setExposure(mesh->getExposure()+0.05);
    } else if (key == '-') {
        if (mesh->getExposure() > 0.05) {
            mesh->setExposure(mesh->getExposure()-0.05);
        }
    } else if (key == 'm') {
        currprog = (currprog+1)%NUM_PROGS;
    } else if (key == ' ') {
        if (pngFilename.length()) shouldWritePngFile = true;
        if (exrFilename.length()) shouldWriteExrFile = true;
        if (plyFilename.length()) shouldWritePlyFile = true;
    } else if (key == ']') {
        planemanager->movePlane(0.01);
    } else if (key == '[') {
        planemanager->movePlane(-0.01);
    } else if (key == 'p') {
        planemanager->togglePlane();
    } else if (key == 'h') {
        // cout << helpstring << endl;
    }
}

// Trackball vars
int ox, oy;
bool moving;
float lastquat[4];
float curquat[4];
float camdist;
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        moving = true;
        ox = x;
        oy = y;
    } else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
        moving = false;
    } else if(button == 3) { // Mouse wheel up
        camdist *= 1.1;
    } else if(button == 4) { // Mouse wheel down
        camdist /= 1.1;
    }
}
void mousemove(int x, int y) {
    if (moving) {
        trackball(lastquat,
            (2*ox-width3d)/(float)width3d,
            (height3d-2*oy)/(float)height3d,
            (2*x-width3d)/(float)width3d,
            (height3d-2*y)/(float)height3d
        );
        ox = x;
        oy = y;
        add_quats(lastquat, curquat, curquat);
    }
}

void draw3D() {
    // Setup camera
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    GLfloat m[4][4];
    glTranslatef(0,0,-camdist);
    build_rotmatrix(m, curquat);
    glMultMatrixf(&m[0][0]);
    glTranslatef(-0.5,-0.5,-0.5);

    mesh->Render();
    mesh->RenderLights();
    // Draw plane
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, tex);
    if (currprog == PROG_SOURCEMAP || currprog == PROG_MEDIALAXIS) {
        // FIXME: Not correct for 3D!!!
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glUniform1i(glGetUniformLocation(progs[currprog], "maxidx"), cudamap.n);
    } else {
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    planemanager->Render();
    glBindTexture(GL_TEXTURE_3D, 0);
    glBindVertexArray(0);
    glPopMatrix();
}

void drawField() {
    if (shouldWritePlyFile || shouldWriteExrFile) {
        if (shouldWritePlyFile) {
            outputPLY(plyFilename.c_str(), distancefield, width, height, NULL);
            shouldWritePlyFile = false;
        }
        if (shouldWriteExrFile) {
            outputEXR(exrFilename.c_str(), distancefield, width, height, 2);
            shouldWriteExrFile = false;
        }
    }
    glUseProgram(progs[currprog]);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, tex);
    //glBindTexture(GL_TEXTURE_BUFFER,tbo_tex);
    //glTexBuffer(GL_TEXTURE_BUFFER,GL_R32F,pbo);
    if (currprog == PROG_SOURCEMAP || currprog == PROG_MEDIALAXIS) {
        // FIXME: Not correct for 3D!!!
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glUniform1i(glGetUniformLocation(progs[currprog], "maxidx"), cudamap.n);
    } else {
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    glUniform1f(glGetUniformLocation(progs[currprog], "exposure"), exposure);

    Eigen::Vector3f planePoint = planemanager->planePoint;
    Eigen::Vector3f planeNormal = planemanager->planeNormal;
    Eigen::Vector3f planeAxis = planemanager->planeAxis;
    Eigen::Vector3f yax = planeNormal.cross(planeAxis);
    glUniform3f(glGetUniformLocation(progs[currprog], "pt"), planePoint[0], planePoint[1], planePoint[2]);
    glUniform3f(glGetUniformLocation(progs[currprog], "xax"), planeAxis[0], planeAxis[1], planeAxis[2]);
    glUniform3f(glGetUniformLocation(progs[currprog], "yax"), yax[0], yax[1], yax[2]);

    glDrawArrays(GL_TRIANGLES,0,6);
    glBindTexture(GL_TEXTURE_3D, 0);
    glBindVertexArray(0);
    if (shouldWritePngFile) {
        glReadPixels(width3d,0,width, height, GL_RGB, GL_UNSIGNED_BYTE, (void*) imagedata);
        outputPNG(pngFilename.c_str(), imagedata, width, height);
        shouldWritePngFile = false;
    }
    if (shouldExitImmediately) {
        exit(0);
    }
}

void draw() {
    glClearColor(0.1f,0.1f,0.2f,1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(width3d,0,width,height);
    drawField();
    glViewport(0,0,width3d,height3d);
    draw3D();
    glutSwapBuffers();
}

void initRenderTextures() {
    glGenTextures(1, &rfr_tex);
    glBindTexture(GL_TEXTURE_2D, rfr_tex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenRenderbuffers(1, &rfr_fbo_z);
    glBindRenderbuffer(GL_RENDERBUFFER, rfr_fbo_z);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glGenFramebuffers(1, &rfr_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, rfr_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rfr_tex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rfr_fbo_z);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
    glUniform1i(glGetUniformLocation(progs[n], "threshold"), 8);
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
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width+width3d, max(height, height3d));
    glutCreateWindow("Light Localization");
    glutDisplayFunc(draw);
    glutIdleFunc(draw);
    glutKeyboardFunc(keydown);
    glutMouseFunc(mouse);
    glutMotionFunc(mousemove);
    openglInit();

    // -=-=-= SETUP OTHER STUFF=-=-=-
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45,width3d/(float)height3d, 0.001f, 100.f);
    trackball(curquat, 0.0, 0.0, 0.0, 0.0);
    camdist = 4;
    exposure = 10;

    imagedata = new unsigned char[3*width*height];
    distancefield = new float[2*width*width*width];

    initRenderTextures();
    setupFullscreenQuad();

    setupProg("tboshader.f.glsl",PROG_ID);
    setupProg("sourcemap.f.glsl",PROG_SOURCEMAP);
    setupProg("medialaxis.f.glsl",PROG_MEDIALAXIS);
    currprog = 0;

    int dim = 256;
    if (options[INPUT_SCENEFILE]) {
        mesh = new LitMesh(&cudamap);
        mesh->ReadFromPly(options[INPUT_SCENEFILE].arg);
    } else {
        option::printUsage(cout, usage);
        return 0;
    }

    if (options[EXIT_IMMEDIATELY]) {
        shouldExitImmediately = true;
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
    planemanager = new PlaneManager();
    planemanager->setExposure(exposure);

    if (options[INPUT_VOLUME]) {
        ifstream in(options[INPUT_VOLUME].arg, ios::in | ios::binary);
        in.read((char*) &dim, sizeof(int));
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                in.read((char*) &(distancefield[2*(i*dim*dim + j*dim)]), sizeof(float)*2*dim);
            }
        }
        in.close();
        mesh->cudaInit(dim);
    } else {
        mesh->cudaInit(dim);
        cout << "Computing field..." << endl;
        Cudamap_compute(&cudamap, distancefield);
        cout << "Done." << endl;
        Cudamap_free(&cudamap);
    }
    if (options[OUTPUT_VOLUME]) {
        ofstream out(options[OUTPUT_VOLUME].arg, ios::out | ios::trunc | ios::binary);
        out.write((char*) &dim, sizeof(int));
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                out.write((char*) &(distancefield[2*(i*dim*dim + j*dim)]), sizeof(float)*2*dim);
            }
        }
        out.close();
    }
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_3D, tex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, dim, dim, dim, 0, GL_RG, GL_FLOAT, distancefield);
    glBindTexture(GL_TEXTURE_3D, 0);

    glutMainLoop();
}
