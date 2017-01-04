#ifndef _PLANE_MANAGER_H
#define _PLANE_MANAGER_H

#include "opengl_compat.h"
#include <Eigen/Eigen>
#include <vector>

class PlaneManager {
    public:
        PlaneManager();
        void movePlane(float amount);
        void togglePlane();
        void Render();
        void setExposure(float e) { exposure = e; }

        const float* point() const { return planePoint.data(); }
        const float* normal() const { return planeNormal.data(); }
        const float* axis() const { return planeAxis.data(); }
    private:
        void initShaders();
        Eigen::Vector3f planePoint, planeNormal, planeAxis;

        std::vector<Eigen::Vector3f> default_points;
        std::vector<Eigen::Vector3f> default_normals;
        std::vector<Eigen::Vector3f> default_axes;
        int planeidx;

        float exposure;
        GLuint progid;
        GLuint expuniform, puniform, xaxuniform, yaxuniform;
        GLuint mvmatrixuniform, projectionmatrixuniform;
};
#endif
