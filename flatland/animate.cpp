#include "fileio.h"
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

using namespace std;

bool endswith(const string& s, string e) {
    if (s.length() > e.length())
        return s.compare(s.length()-e.length(), e.length(), e) == 0;
    else
        return false;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: animate image.png output%02d.png [guideimage.exr]" << endl;
        return 0;
    }

    string imagefilename, guidefilename, outputpattern;
    imagefilename = argv[1];
    outputpattern = argv[2];
    if (argc < 4) guidefilename = imagefilename;
    else guidefilename = argv[3];

    unsigned char* image;
    float* guideimage;
    int w, h;
    cout << imagefilename << endl;
    readPngImage(imagefilename.c_str(), &image, w, h);
    if (endswith(guidefilename, ".exr")) {
        readExrImage(guidefilename.c_str(), &guideimage, w, h, 3);
    //} else if (endswith(guidefilename, ".png")) {
        //readPngImage(guidefilename.c_str(), &guideimage, w, h, 3);
    } else {
        cout << "Unknown file extension for guide image!" << endl;
        return 1;
    }

    vector<pair<float,int> > order;
    for (int i = 0; i < w*h; i++) {
        if (isnan(guideimage[3*i]) || isinf(guideimage[3*i])) order.push_back(make_pair(0, i));
        else order.push_back(make_pair(guideimage[3*i], i));
    }
    cout << "Sorting..." << endl;
    sort(order.begin(), order.end());
    cout << "Done sorting..." << endl;
    float step = order.back().first/150;
    cout << order.back().first << " " << step << endl;

    unsigned char* outputimage = new unsigned char[3*w*h];
    memset(outputimage, 0, 3*w*h);
    float lastoutput = 0;
    int n = 0;
    float last = -1;
    float curr;
    for (int i = 0; i < order.size(); i++) {
        int idx = order[i].second;
        curr = order[i].first;
        for (int j = 0; j < 3; j++) {
            outputimage[3*idx + j] = image[3*idx + j];
        }
        if (curr - lastoutput > step || i == order.size()-1) {
            lastoutput = curr;
            char tmp[50];
            sprintf(tmp, outputpattern.c_str(), n++);
            outputPNG(tmp, outputimage, w, h);
            cout << n << ": Outputting at " << lastoutput << endl;
        }
    }
    return 0;
}
