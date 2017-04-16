#ifndef __VISUAL_OPTIONS
#define __VISUAL_OPTIONS
#include "optionparser.h"
#include <cstdio>
enum optionIndex {
    UNKNOWN,
    INPUT_SCENEFILE,
    INPUT_STDIN,
    RESOLUTION,
    DISPLAYSCALE,
    MODE,
    EXIT_IMMEDIATELY,
    OUTPUT_IMAGEFILE,
    OUTPUT_MESHFILE,
    PRINT_SUCCESS,
    //NOISE_NORMAL,
    //NOISE_GEO,
    NOISE_INTENSITY
};

option::ArgStatus filename(const option::Option& opt, bool msg) {
    if (opt.arg) return option::ARG_OK;
    else {
        if (msg) fprintf(stderr, "Error: filename required\n");
        return option::ARG_ILLEGAL;
    }
}
option::ArgStatus integer(const option::Option& opt, bool msg) {
    if (atoi(opt.arg) > 0) return option::ARG_OK;
    else {
        if (msg) fprintf(stderr, "Error: Invalid resolution\n");
        return option::ARG_ILLEGAL;
    }
}

option::ArgStatus real(const option::Option& opt, bool msg) {
    if (atof(opt.arg) > 0) return option::ARG_OK;
    else return option::ARG_ILLEGAL;
}

const option::Descriptor usage[] =
{
    {UNKNOWN,          0, "",  "",          option::Arg::None, "Usage: visual [options]\n\n"
                                                               "Options:"},
    {INPUT_SCENEFILE,  0, "i", "scene",     filename,          "  -i, --scene scene.scn\tUse scene and lights specified in scene.scn"},
    {INPUT_STDIN,      0, "", "from-stdin", option::Arg::None, "  --from-stdin\tUse scene and lights specified in standard input"},
    {RESOLUTION,       0, "rw", "res",      integer,           "  -w, --res resolution\tCompute fields at this resolution"},
    {DISPLAYSCALE,     0, "s", "scale",     integer,           "  -s, --scale displayscale\tShow images at scale*res resolution"},
    {MODE,             0, "m", "mode",      integer,           "  -m, --mode modeindex\tVisualization mode (0: distance field, 1: sourcemap, 2: voronoi, 3: medial axis, 4: density)"},
    {EXIT_IMMEDIATELY, 0, "q", "",          option::Arg::None, "  -q                             \tExit immediately after output"},
    {OUTPUT_IMAGEFILE, 0, "o", "imagefile", filename,          "  -o img.ppm, --imagefile img.ppm\t"},
    {OUTPUT_MESHFILE,  0, "",  "meshfile",  filename,          "  --meshfile mesh.ply            \t"},
    {PRINT_SUCCESS,    0, "",  "print-success",  option::Arg::None, "  --print-success                \t"},
    {NOISE_INTENSITY,  0, "", "noise",      real,             "  --noise noiseamount\tPercent magnitude of intensity measurement noise"},
    {0,0,0,0,0,0}
};
#endif
