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
        void Render(GLuint progid);
        void setExposure(float e) { exposure = e; }
        float getExposure() const { return exposure; }

        const float* point() const { return planePoint.data(); }
        const float* normal() const { return planeNormal.data(); }
        const float* axis() const { return planeAxis.data(); }

        Eigen::Vector3f planePoint, planeNormal, planeAxis;
    private:
        std::vector<Eigen::Vector3f> default_points;
        std::vector<Eigen::Vector3f> default_normals;
        std::vector<Eigen::Vector3f> default_axes;
        int planeidx;
        float exposure;
};
#endif
