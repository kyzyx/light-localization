#ifndef __VISUAL_OPTIONS
#define __VISUAL_OPTIONS
#include "optionparser.h"
#include <cstdio>
enum optionIndex {
    UNKNOWN,
    INPUT_SCENEFILE,
    INPUT_VOLUME,
    RESOLUTION,
    MODE,
    EXIT_IMMEDIATELY,
    OUTPUT_VOLUME,
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
    {INPUT_VOLUME,     0, "v", "volume",    filename,          "  -v, --volume volume.vlm\tUse precomputed field specified in volume.vlm"},
    {RESOLUTION,       0, "rw", "res",       integer,          "  -w, --res resolution\tCompute fields at this resolution"},
    {MODE,             0, "m", "mode",       integer,          "  -m, --mode modeindex\tVisualization mode (0: distance field, 1: sourcemap, 2: medial axis)"},
    {EXIT_IMMEDIATELY, 0, "q", "",          option::Arg::None, "  -q                             \tExit immediately after output"},
    {OUTPUT_VOLUME,    0, "",  "outputvolume", filename,          "  --outputvolume volume.vlm      \t"},
    {OUTPUT_IMAGEFILE, 0, "o", "imagefile",  filename,          "  -o img.ppm, --imagefile img.ppm\t"},
    {OUTPUT_MESHFILE,  0, "",  "meshfile",   filename,          "  --meshfile mesh.ply            \t"},
    {0,0,0,0,0,0}
};
#endif
