#include "loadshader.h"
#include "planemanager.h"

PlaneManager::PlaneManager() {
    exposure = 10;
    planeidx = -1;
    default_normals.push_back(Eigen::Vector3f(1,0,0));
    default_normals.push_back(Eigen::Vector3f(0,1,0));
    default_normals.push_back(Eigen::Vector3f(0,0,1));
    default_points.push_back(Eigen::Vector3f(0.5,0,0));
    default_points.push_back(Eigen::Vector3f(0,0.5,0));
    default_points.push_back(Eigen::Vector3f(0,0,0.5));
    default_axes.push_back(Eigen::Vector3f(0,1,0));
    default_axes.push_back(Eigen::Vector3f(0,0,1));
    default_axes.push_back(Eigen::Vector3f(1,0,0));
    togglePlane();
}

void PlaneManager::movePlane(float amount) {
    planePoint += amount*planeNormal;
}

void PlaneManager::togglePlane() {
    planeidx++;
    planeidx %= default_normals.size();
    planePoint = default_points[planeidx];
    planeNormal = default_normals[planeidx];
    planeAxis = default_axes[planeidx];
}

void PlaneManager::Render(GLuint progid) {
    GLfloat modelview[16];
    GLfloat projection[16];
    Eigen::Vector3f yax = planeNormal.cross(planeAxis);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glUseProgram(progid);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
    glGetFloatv(GL_PROJECTION_MATRIX, projection);
    glUniformMatrix4fv(glGetUniformLocation(progid, "projectionmatrix"), 1, GL_FALSE, projection);
    glUniformMatrix4fv(glGetUniformLocation(progid, "modelviewmatrix"), 1, GL_FALSE, modelview);
    glUniform1f(glGetUniformLocation(progid, "exposure"), exposure);
    glUniform3f(glGetUniformLocation(progid, "pt"), planePoint[0], planePoint[1], planePoint[2]);
    glUniform3f(glGetUniformLocation(progid, "xax"), planeAxis[0], planeAxis[1], planeAxis[2]);
    glUniform3f(glGetUniformLocation(progid, "yax"), yax[0], yax[1], yax[2]);
    glDrawArrays(GL_TRIANGLES,0,6);
}
