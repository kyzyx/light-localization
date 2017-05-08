#include "solveCeres.h"
#include <ceres/ceres.h>
#include <iostream>
#define DIM 3

using namespace ceres;
using namespace std;

struct CostFunctor {
    CostFunctor(double* g, double i, int numlights) {
        for (int j = 0; j < DIM; j++) {
            p[j] = g[j];
            n[j] = g[DIM+j];
        }
        intensity = i;
        nl = numlights;
    }
    template <typename T>
        bool operator()(const T* const l, const T* const intensities, T* residual) const {
            residual[0] = T(intensity);
            for (int i = 0; i < nl; i++) {
                T LdotL = T(0);
                T ndotLn = T(0);
                for (int j = 0; j < DIM; j++) {
                    T L = l[DIM*i+j] - T(p[j]);
                    ndotLn += T(n[j])*L;
                    LdotL += L*L;
                }
                LdotL = ceres::sqrt(LdotL);
                residual[0] -= intensities[i]*ndotLn/(LdotL*LdotL*LdotL);
            }
            return true;
        }

    int nl;
    double p[DIM];
    double n[DIM];
    double intensity;
};

CostFunction* Create(double* g, double* i, int numlights) {
    if (numlights == 1) {
        return new AutoDiffCostFunction<CostFunctor, 1, DIM, 1>(new CostFunctor(g, *i, numlights));
    } else if (numlights == 2) {
        return new AutoDiffCostFunction<CostFunctor, 1, 2*DIM, 2>(new CostFunctor(g, *i, numlights));
    } else if (numlights == 3) {
        return new AutoDiffCostFunction<CostFunctor, 1, 3*DIM, 3>(new CostFunctor(g, *i, numlights));
    } else if (numlights == 4) {
        return new AutoDiffCostFunction<CostFunctor, 1, 4*DIM, 4>(new CostFunctor(g, *i, numlights));
    } else if (numlights == 5) {
        return new AutoDiffCostFunction<CostFunctor, 1, 5*DIM, 5>(new CostFunctor(g, *i, numlights));
    } else if (numlights == 6) {
        return new AutoDiffCostFunction<CostFunctor, 1, 6*DIM, 6>(new CostFunctor(g, *i, numlights));
    } else if (numlights == 7) {
        return new AutoDiffCostFunction<CostFunctor, 1, 7*DIM, 7>(new CostFunctor(g, *i, numlights));
    } else if (numlights == 8) {
        return new AutoDiffCostFunction<CostFunctor, 1, 8*DIM, 8>(new CostFunctor(g, *i, numlights));
    } else if (numlights == 9) {
        return new AutoDiffCostFunction<CostFunctor, 1, 9*DIM, 9>(new CostFunctor(g, *i, numlights));
    } else if (numlights == 10) {
        return new AutoDiffCostFunction<CostFunctor, 1, 10*DIM, 10>(new CostFunctor(g, *i, numlights));
    } else {
        cerr << "Error: unhandled number of lights " << numlights << endl;
    }
}

double solveIntensitiesCeres(
        double* geometry, double* intensities, int n,
        double* lightparams, double* lightintensities, int numlights
        )
{
    Problem problem;
    for (int i = 0; i < n; i++) {
        problem.AddResidualBlock(
                Create(geometry+i*DIM*2, intensities+i, numlights),
                NULL, lightparams, lightintensities);
    }
    problem.SetParameterBlockConstant(lightparams);
    for (int i = 0; i < numlights; i++) {
        problem.SetParameterLowerBound(lightintensities, i, 0.);
    }
    Solver::Options options;
    options.linear_solver_type = DENSE_QR;
    options.logging_type = SILENT;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    return summary.final_cost;
}

double solveCeres(
        double* geometry, double* intensities, int n,
        double* lightparams, double* lightintensities, int numlights
        )
{
    Problem problem;
    for (int i = 0; i < n; i++) {
        problem.AddResidualBlock(
                Create(geometry+i*DIM*2, intensities+i, numlights),
                NULL, lightparams, lightintensities);
    }
    for (int i = 0; i < numlights; i++) {
        for (int j = 0; j < DIM; j++) {
            problem.SetParameterLowerBound(lightparams, i*DIM+j, -.99);
            problem.SetParameterUpperBound(lightparams, i*DIM+j, 0.99);
        }
        problem.SetParameterLowerBound(lightintensities, i, 0.);
    }
    Solver::Options options;
    options.linear_solver_type = DENSE_QR;
    options.logging_type = SILENT;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    return summary.final_cost;
}
