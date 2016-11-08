#ifndef __FILEIO_H
#define __FILEIO_H

bool outputPNG(const char* filename, const unsigned char* image, int width, int height, int channels=3);
void outputPLY(const char* filename, float* data, int width, int height, float* colors=0);

#endif
