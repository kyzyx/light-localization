#include "opengl_compat.h"
#include <vector>
#include <iostream>

#include "loadshader.h"
#include "geometry.h"
#include "cudamap.h"

const int width = 600;
const int height = 600;

using namespace std;
using namespace Eigen;

class Scene {
    public:
        Scene() : minp(Vector2f(0,0)), maxp(Vector2f(0,0)) { }
        ~Scene() { Cudamap_free(&cm); }
        void addSegment(Line l) {
            extendBbox(l.p1);
            extendBbox(l.p2);
            lines.push_back(l);
        }
        int intersectsAny(Line a) {
            for (int i = 0; i < lines.size(); i++) {
                if (intersects(lines[i],a)) {
                    return i+1;
                }
            }
            return 0;
        }
        void rasterize(int w, int h) {
            for (int i = 0; i < lines.size(); i++) {
                Line l = lines[i];
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
                    lineid.push_back(i);
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
        }
        void initCuda(int w, int h) {
            cm.w = w;
            cm.h = h;
            field = new float[w*h];
            cm.maxx = maxp[0];
            cm.maxy = maxp[1];
            cm.minx = minp[0];
            cm.miny = minp[1];
            cm.n = surfels.size()/4;
            Cudamap_init(&cm, surfels.data());
        }
        void setCudaGLTexture(GLuint pbo) {
            Cudamap_setGLTexture(&cm, pbo);
        }

        int numLights() const {
            return lights.size();
        }
        Vector3f getLight(int idx) const {
            return lights[idx];
        }
        void changeIntensity(int i, float intensity) {
            Cudamap_addLight(&cm, intensity - lights[i][2], lights[i][0], lights[i][1]);
            lights[i][2] = intensity;
            computeField();
        }
        void moveLight(float x, float y, int i) {
            Cudamap_addLight(&cm, -lights[i][2], lights[i][0], lights[i][1]);
            lights[i][0] = x;
            lights[i][1] = y;
            Cudamap_addLight(&cm, lights[i][2], lights[i][0], lights[i][1]);
            computeField();
        }
        void deleteLight(int i) {
            Cudamap_addLight(&cm, -lights[i][2], lights[i][0], lights[i][1]);
            lights.erase(lights.begin()+i);
            computeField();
        }
        void addLight(float x, float y, float intensity=1) {
            Cudamap_addLight(&cm, intensity, x, y);
            lights.push_back(Vector3f(x,y,intensity));
            computeField();
        }
        void computeField() {
            Cudamap_compute(&cm, field);
        }
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

    protected:
        void extendBbox(Vector2f p) {
            minp[0] = min(minp[0], p[0]);
            minp[1] = min(minp[1], p[1]);
            maxp[0] = max(maxp[0], p[0]);
            maxp[1] = max(maxp[1], p[1]);
        }

        Vector2f minp, maxp;
        vector<Line> lines;
        vector<Vector3f> lights;

        Cudamap cm;
        float* field;
        vector<float> surfels;
        vector<int> coords;
        vector<int> lineid;
        vector<float> intensities;
};

Scene s;
GLuint vao;
GLuint vbo[2];
GLuint pbo, tbo_tex, progid;

int selectedlight = -1;
int dragging = 0;
Vector2f offset;

void keydown(unsigned char key, int x, int y) {
    if (key == ',') {
        GLuint loc = glGetUniformLocation(progid, "exposure");
        float exposure;
        glGetUniformfv(progid, loc, &exposure);
        if (exposure > 0.05)
            glUniform1f(loc, exposure-0.05);
    }
    else if (key == '.') {
        GLuint loc = glGetUniformLocation(progid, "exposure");
        float exposure;
        glGetUniformfv(progid, loc, &exposure);
        glUniform1f(loc, exposure+0.05);
    } else if (key == 127 && selectedlight >= 0) {
        s.deleteLight(selectedlight);
        selectedlight = -1;
        dragging = 0;
    }
}

const float RADIUS = 0.05;
void click(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        Vector2f p = s.clip2world(x,height-y,width,height);
        selectedlight = -1;
        for (int i = 0; i < s.numLights(); i++) {
            if ((p-s.getLight(i).head(2)).squaredNorm() < RADIUS*RADIUS) {
                selectedlight = i;
                break;
            }
        }
        if (selectedlight < 0) {
            selectedlight = s.numLights();
            s.addLight(p[0], p[1]);
        }
        dragging = 1;
        offset = s.getLight(selectedlight).head(2) - p;
    } else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
        dragging = 0;
    }
}
void draw() {
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER,tbo_tex);
    glTexBuffer(GL_TEXTURE_BUFFER,GL_R32F,pbo);
    glDrawArrays(GL_TRIANGLES,0,6);
    glBindTexture(GL_TEXTURE_BUFFER,0);
    glBindVertexArray(0);
    glutSwapBuffers();
}

void setupWindow(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("Light Localization");
    glutDisplayFunc(draw);
    glutIdleFunc(draw);
    glutKeyboardFunc(keydown);
    glutMouseFunc(click);
    openglInit();
}

int main(int argc, char** argv) {
    setupWindow(argc, argv);
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glGenTextures(1, &tbo_tex);
    glBindTexture(GL_TEXTURE_BUFFER, tbo_tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, pbo);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    ShaderProgram* prog;
    prog = new FileShaderProgram("tboshader.v.glsl", "tboshader.f.glsl");
    prog->init();
    delete prog;
    progid = prog->getProgId();
    glUseProgram(progid);
    glUniform1i(glGetUniformLocation(progid, "buffer"), 0);
    glUniform2i(glGetUniformLocation(progid, "dim"), width, height);
    glUniform1f(glGetUniformLocation(progid, "exposure"), 0.5);

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

    s.addSegment(Line(Vector2f(0, -0.01), Vector2f(0, 1.01)));
    s.addSegment(Line(Vector2f(-0.01, 1), Vector2f(1.01, 1)));
    s.addSegment(Line(Vector2f(1, 1.01), Vector2f(1, -0.01)));
    s.addSegment(Line(Vector2f(1.01, 0), Vector2f(-0.01, 0)));
    s.rasterize(width, height);

    s.initCuda(width, height);
    s.setCudaGLTexture(pbo);

    s.addLight(0.5, 0.5);

    glutMainLoop();
}