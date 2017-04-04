#ifndef _SOLVE_CERES
#define _SOLVE_CERES
double solveIntensitiesCeres(
        double* geometry, double* intensities, int n,
        double* lightparams, double* lightintensities, int numlights
        );
double solveCeres(
        double* geometry, double* intensities, int n,
        double* lightparams, double* lightintensities, int numlights
        );
#endif
