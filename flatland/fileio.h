#ifndef __FILEIO_H
#define __FILEIO_H

bool outputEXR(const char* filename, const float* image, int width, int height, int channels=3);
bool outputPNG(const char* filename, const unsigned char* image, int width, int height, int channels=3);
void outputPLY(const char* filename, float* data, int width, int height, float* colors=0);

bool readExrImage(const char* filename,
        float** image,
        int& width,
        int& height,
        int channels,
        bool preallocated=false);
bool readPngImage(const char* filename,
        unsigned char** image,
        int& width,
        int& height,
        bool preallocated=false);
#endif
