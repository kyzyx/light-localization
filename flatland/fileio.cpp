#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfHeader.h>
#include <png.h>
#include <cstdlib>
#include <fstream>

using namespace Imf;
using namespace Imath;

bool outputEXR(const char* filename,
        const float* image,
        int width,
        int height,
        int channels)
{
    Rgba* pixels = new Rgba[width*height];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pidx = j + i*width;
            int iidx = j + (height-i-1)*width;
            pixels[pidx].r = image[channels*iidx];
            if (channels > 2) {
                pixels[pidx].g = image[channels*iidx+1];
                pixels[pidx].b = image[channels*iidx+2];
            } else {
                pixels[pidx].g = image[channels*iidx];
                pixels[pidx].b = image[channels*iidx];
            }
            if (channels > 3) {
                pixels[pidx].a = image[channels*iidx+3];
            } else {
                pixels[pidx].a = 1;
            }
        }
    }
    RgbaOutputFile file(filename, width, height, WRITE_RGBA);
    file.setFrameBuffer(pixels, 1, width);
    file.writePixels(height);
    delete pixels;
    return true;
}

bool outputPNG(const char* filename,
        const unsigned char* image,
        int width,
        int height,
        int channels)
{
    static int filenum = 0;

    png_structp png_ptr;
    png_infop info_ptr;
    unsigned int sig_read = 0;
    int color_type, interlace_type;
    png_bytep * row_pointers;

    char* fmtfilename = new char[strlen(filename) + 4];
    sprintf(fmtfilename, filename, filenum++);
    FILE *fp;
    if ((fp = fopen(fmtfilename, "wb")) == NULL) {
        delete [] fmtfilename;
        return false;
    }
    delete [] fmtfilename;

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return false;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fclose(fp);
        return false;
    }

    png_init_io(png_ptr, fp);

    if (setjmp(png_jmpbuf(png_ptr))) {
        fclose(fp);
        return false;
    }

    png_set_IHDR(png_ptr, info_ptr, width, height,
            8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    if (setjmp(png_jmpbuf(png_ptr))) {
        fclose(fp);
        return false;
    }

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    unsigned int row_bytes = png_get_rowbytes(png_ptr,info_ptr);
    const unsigned char* im = image;
    for (int i = 0; i < height; i++) {
        row_pointers[i] = (png_byte*) malloc(row_bytes);
        if (channels == 3) {
            memcpy(row_pointers[i], im + (row_bytes * (height-1-i)), row_bytes);
        } else {
            for (int j = 0; j < width; ++j) {
                int idx = j + width*(height-1-i);
                for (int k = 0; k < 3; ++k)
                    row_pointers[i][3*j+k] = k<channels?im[channels*idx+k]:im[channels*idx+channels-1];
            }
        }
    }

    png_write_image(png_ptr, row_pointers);

    if (setjmp(png_jmpbuf(png_ptr))) {
        fclose(fp);
        return false;
    }

    png_write_end(png_ptr, NULL);

    for (int y=0; y<height; y++)
        free(row_pointers[y]);
    free(row_pointers);

    fclose(fp);
    return true;
}

int f2c(float f) {
    return std::min(std::max(f*255,0.f),255.f);
}

void outputPLY(const char* filename, float* data, int width, int height, int channels, float* colors) {
    static int filenum = 0;
    char* fmtfilename = new char[strlen(filename) + 4];
    sprintf(fmtfilename, filename, filenum++);
    std::ofstream out(fmtfilename);
    delete [] fmtfilename;
    float maxval = 0.01;
    for (int i = 0; i < width*height; i++) {
        if (data[channels*i] >= 1e9) data[channels*i] = 0;
        maxval = std::max(maxval, data[channels*i]);
    }
    maxval *= 2;
    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << (width*height) << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    if (colors) out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "element face " << (2*(width-1)*(height-1)) << std::endl;
    out << "property list uchar int vertex_indices" << std::endl;
    out << "end_header" << std::endl;
    for (int i = 0; i < width*height; i++) {
        out << (i%width) << " " << (i/width) << " " << width*data[channels*i]/maxval;
        //if (colors) out << f2c(colors[3*i]) << " " <<  f2c(colors[3*i+1]) << " " <<  f2c(colors[3*i+2]) << std::endl;
        if (colors) {
            if (colors[i] > 0) out << " " << (255-f2c(colors[i])) << " 0 0";
            else out << " 200 200 200";
        }
        out << std::endl;
    }
    for (int i = 0; i < (width-1)*(height-1); i++) {
        int x = i%(width-1);
        int y = i/(width-1);
        int idx = x + y*width;
        out << 3 << " " << idx << " " << (idx + 1) << " " << (idx + 1 + width) << std::endl;
        out << 3 << " " << idx << " " << (idx + 1 + width) << " " << (idx + width) << std::endl;
    }
}

bool readExrImage(const char* filename,
        float** image,
        int& width,
        int& height,
        int channels,
        bool preallocated)
{
    RgbaInputFile f(filename);
    Box2i dw = f.dataWindow();
    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    Array2D<Rgba> pixels;
    pixels.resizeErase(height, width);
    f.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
    f.readPixels(dw.min.y, dw.max.y);
    if (!preallocated) *image = new float[channels*width*height];
    float* im = *image;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = width*(height-i-1) + j;
            im[3*idx] = pixels[i][j].r;
            if (channels > 1) im[3*idx+1] = pixels[i][j].g;
            if (channels > 2) im[3*idx+2] = pixels[i][j].b;
            if (channels > 3) im[3*idx+3] = pixels[i][j].a;
        }
    }
    return true;
}

bool readPngImage(const char* filename,
        unsigned char** image,
        int& width,
        int& height,
        bool preallocated)
{
    png_structp png_ptr;
    png_infop info_ptr;
    unsigned int sig_read = 0;
    int color_type, interlace_type;
    FILE *fp;

    if ((fp = fopen(filename, "rb")) == NULL)
        return false;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        fclose(fp);
        return false;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fclose(fp);
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, sig_read);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_ALPHA | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND, NULL);
    int bit_depth;
    png_uint_32 pngw, pngh;
    png_get_IHDR(png_ptr, info_ptr, &pngw, &pngh, &bit_depth, &color_type,
            &interlace_type, NULL, NULL);
    width = pngw;
    height = pngh;

    unsigned int row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    if (!preallocated) *image = new unsigned char[width*height*3];
    char* im = (char*) *image;

    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);

    for (int i = 0; i < height; i++) {
        memcpy(im+(row_bytes * (height-1-i)), row_pointers[i], row_bytes);
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    return true;
}
