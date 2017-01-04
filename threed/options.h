#ifndef __VISUAL_OPTIONS
#define __VISUAL_OPTIONS
#include "optionparser.h"
#include <cstdio>
enum optionIndex {
    UNKNOWN,
    INPUT_SCENEFILE,
    RESOLUTION,
    DISPLAYSCALE,
    EXIT_IMMEDIATELY,
    OUTPUT_IMAGEFILE,
    OUTPUT_MESHFILE,
    //NOISE_NORMAL,
    //NOISE_GEO,
    //NOISE_INTENSITY,
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

const option::Descriptor usage[] =
{
    {UNKNOWN,          0, "",  "",          option::Arg::None, "Usage: visual [options]\n\n"
                                                               "Options:"},
    {INPUT_SCENEFILE,  0, "i", "scene",     filename,          "  -i, --scene scene.scn\tUse scene and lights specified in scene.scn"},
    {RESOLUTION,       0, "rw", "res",       integer,          "  -w, --res resolution\tCompute fields at this resolution"},
    {DISPLAYSCALE,     0, "s", "scale",      integer,          "  -s, --scale displayscale\tShow images at scale*res resolution"},
    {EXIT_IMMEDIATELY, 0, "q", "",          option::Arg::None, "  -q                             \tExit immediately after output"},
    {OUTPUT_IMAGEFILE, 0, "o", "imagefile", filename,          "  -o img.ppm, --imagefile img.ppm\t"},
    {OUTPUT_MESHFILE,  0, "",  "meshfile",  filename,          "  --meshfile mesh.ply            \t"},
    {0,0,0,0,0,0}
};
#endif
