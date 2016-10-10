#ifndef __FLATLAND_GEOMETRY
#define __FLATLAND_GEOMETRY
#include <Eigen/Eigen>

class Line {
    public:
        Line()
            : p1(Eigen::Vector2f(0,0)), p2(Eigen::Vector2f(0,0)) {}
        Line(Eigen::Vector2f a, Eigen::Vector2f b)
            : p1(a), p2(b) {}
        Eigen::Vector2f p1;
        Eigen::Vector2f p2;

        Eigen::Vector2f vec() const {
            return (p2-p1).normalized();
        }
        Eigen::Vector2f normal() const {
            Eigen::Vector2f v = (p2-p1).normalized();
            return Eigen::Vector2f(v[1], -v[0]);
        }
};

float ccw(Eigen::Vector2f a, Eigen::Vector2f b, Eigen::Vector2f c) {
    return (b[0] - a[0])*(c[1] - a[1]) - (c[0] - a[0])*(b[1] - a[1]);
}
bool intersects(Line l1, Line l2) {
    if (ccw(l1.p1, l1.p2, l2.p1)*ccw(l1.p1, l1.p2, l2.p2) > 0) return false;
    if (ccw(l2.p1, l2.p2, l1.p1)*ccw(l2.p1, l2.p2, l1.p2) > 0) return false;
    return true;
}
Eigen::Vector2f projectPointToLine(Eigen::Vector2f point, Line line) {
    Eigen::Vector2f v = line.vec();
    return line.p1 + (point-line.p1).dot(v)*v;
}
#endif
