#include "glplot.h"
#include "loadshader.h"

GLPlot::GLPlot() {
    color[0] = 1;
    color[1] = 0;
    color[2] = 0;

    xscale = 1;
    yscale = 1;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    ShaderProgram* p;
    p = new FileShaderProgram("graph.v.glsl", "graph.f.glsl");
    p->init();
    prog = p->getProgId();
    delete p;

}

void GLPlot::updateData(float* data, int length) {
    this->length = length;
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, length*sizeof(float), data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLPlot::draw() {
    glViewport(vx, vy, vw, vh);
    glBindVertexArray(vao);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "len"), length);
    glUniform2f(glGetUniformLocation(prog, "scale"), xscale, yscale);
    glUniform3f(glGetUniformLocation(prog, "color"), color[0], color[1], color[2]);
    glDrawArrays(GL_LINE_STRIP, 0, length);
    glBindVertexArray(0);
    // FIXME: Draw axes
    // FIXME: Allow offset
}
