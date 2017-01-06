#ifndef _EXTRACT_H
#define _EXTRACT_H
#include <vector>
#include <glm/glm.hpp>


class Extractor {
    public:
        Extractor(float* distancefield, int dim)
            : df(distancefield), w(dim)
        {
            initNeighbors();
        }

        void extract(std::vector<float>& points, float threshold);

    private:
        void initNeighbors();
        glm::vec3 computeDensity(glm::ivec3 stu, float anglethreshold);

        float* df;
        int w;
        std::vector<int> neighbors;
};
#endif
