#ifndef _MESH_H
#define _MESH_H
#include "opengl_compat.h"
#include <vector>
#include "cudamap.h"

class LitMesh {
    public:
        LitMesh(Cudamap* cudamap);
        ~LitMesh();
        void ReadFromPly(const char* filename);
        void Render();
        void RenderLights(float radius=0.01f);

        void cudaInit(int w, int h) {
            cm->w = w;
            cm->h = h;
            cm->n = v.size();
            Cudamap_init(cm, v.data(), n.data());
        }

        void addLight(float intensity, float x, float y, float z) {
            l.push_back(x);
            l.push_back(y);
            l.push_back(z);
            l.push_back(intensity);
            Cudamap_addLight(cm, intensity, x, y, z);
        };

        int NLights() const { return l.size()/4; }
        int NVertices() const { return v.size(); }
        const float* vertices() const { return v.data(); }
        const float* normals() const { return n.data(); }
        const float* lights() const { return l.data(); }
    private:
        void initOpenGL();
        void initShaders();
        void computeLighting();

        Cudamap* cm;
        std::vector<float> v;
        std::vector<float> n;
        std::vector<float> c;
        std::vector<unsigned int> f;
        std::vector<float> l;

        GLuint vao, ibo, meshprogid, lightprogid;
        GLuint vbo[3];
        GLuint meshmvmatrixuniform, meshprojectionmatrixuniform;
        GLuint lightmvmatrixuniform, lightprojectionmatrixuniform;
        GLUquadric* quadric;
};
#endif
