#include <Python.h>
#include "numpy/arrayobject.h"

int SPH_interpolate(double * field, double * comp, const int nx, PyArrayObject *pos, PyArrayObject *radii, PyArrayObject *value, PyArrayObject *weights, const npy_int nval, const int periodic);
