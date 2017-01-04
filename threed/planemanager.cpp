#include "opengl_compat.h"
#include "planemanager.h"

PlaneManager::PlaneManager() {
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

void PlaneManager::Render() {
    Eigen::Vector3f yax = planeNormal.cross(planeAxis);
    Eigen::Vector3f p = planePoint;
    glBegin(GL_TRIANGLES);
        p = planePoint;
        glVertexAttrib2f(1,0.f,0.f);
        glVertex3f(p[0],p[1],p[2]);

        p = planePoint + planeAxis;
        glVertexAttrib2f(1,1.f,0.f);
        glVertex3f(p[0],p[1],p[2]);

        p = planePoint + planeAxis + yax;
        glVertexAttrib2f(1,1.f,1.f);
        glVertex3f(p[0],p[1],p[2]);

        p = planePoint;
        glVertexAttrib2f(1,0.f,0.f);
        glVertex3f(p[0],p[1],p[2]);

        p = planePoint + planeAxis + yax;
        glVertexAttrib2f(1,1.f,1.f);
        glVertex3f(p[0],p[1],p[2]);

        p = planePoint + yax;
        glVertexAttrib2f(1,0.f,1.f);
        glVertex3f(p[0],p[1],p[2]);
    glEnd();
}
