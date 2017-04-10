#ifndef _GL_PLOT_H
#define _GL_PLOT_H
#include "opengl_compat.h"

class GLPlot {
    public:
        GLPlot();

        void setViewport(int x, int y, int w, int h) {
            vx = x;
            vy = y;
            vw = w;
            vh = h;
        }
        void updateData(float* data, int length);
        void draw();

        int getLength() const { return length; }
        void setXScale(double scale) { xscale = scale; }
        void setYScale(double scale) { yscale = scale; }
        void setColor(float r, float g, float b) {
            color[0] = r;
            color[1] = g;
            color[2] = b;
        }
        void setColor(float* c) {
            for (int i = 0; i < 3; i++) color[i] = c[i];
        }

    protected:
        int length;
        float xscale, yscale;
        GLuint vao, vbo;
        GLuint prog;
        float color[3];
        int vx, vy, vw, vh;
};
#endif
