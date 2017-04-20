#ifndef __FILTER_H
#define __FILTER_H
#include <vector>
#include <cmath>

double G(double x, double std) {
    return exp(-(x*x)/(2*std*std))/(sqrt(2*M_PI)*std);
}

template<typename T>
int ConstructKernel(std::vector<T>& kernel, T sigma) {
    int radius = 3*sigma;
    kernel.resize(2*radius+1);
    T tot = 0;
    for (int i = 0; i <= radius; i++) {
        kernel[radius-i] = G(i, sigma);
        kernel[radius+i] = kernel[radius-i];
    }
    for (int i = 0; i < kernel.size(); i++) tot += kernel[i];
    for (int i = 0; i < kernel.size(); i++) kernel[i] /= tot;
    return radius;
}

template<typename T>
void GaussianFilter1D(const std::vector<T>& a, std::vector<T>& out, T sigma) {
    out.clear();
    std::vector<T> kernel;
    int radius = ConstructKernel(kernel, sigma);
    for (int i = 0; i < a.size(); i++) {
        T tot = 0;
        for (int j = 0; j < kernel.size(); j++) {
            tot += kernel[j]*a[(i+j-radius+a.size())%a.size()];
        }
        out.push_back(tot);
    }
}

template<typename T>
void BilateralFilter1D(const std::vector<T>& a, std::vector<T>& out, T domainsigma, T rangesigma) {
    out.clear();
    std::vector<T> kernel;
    int radius = ConstructKernel(kernel, domainsigma);
    for (int i = 0; i < a.size(); i++) {
        T tot = 0;
        T weight = 0;
        for (int j = 0; j < kernel.size(); j++) {
            T d = a[(i+j-radius+a.size())%a.size()] - a[i];
            T currweight = kernel[j]*G(d, rangesigma);
            tot += currweight*a[(i+j-radius+a.size())%a.size()];
            weight += currweight;
        }
        out.push_back(tot/weight);
    }
}

#endif
