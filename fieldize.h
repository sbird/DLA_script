#include <Python.h>
#include "numpy/arrayobject.h"

class SPH_interpolate
{
    public:
        //field: pointer to the array where interpolated data is stored
        //comp: pointer to temporary memory for Kahan summation. Not used if NO_KAHAN is defined
        //nx: size of the above arrays is nx*nx
        //periodic: should we assume input array is periodic?
        SPH_interpolate(double * field_i, double * comp_i, int nx_i, bool periodic_i):
           periodic(periodic_i), nx(nx_i), field(field_i), comp(comp_i)
           {};
        //pos: array of particle positions
        //radii: particule smoothing lengths
        //value: amount to interpolate to grid
        //weights: weights with which to interpolate
        //nval: size of the above arrays
        int do_work(PyArrayObject *pos, PyArrayObject *radii, PyArrayObject *value, PyArrayObject *weights, const npy_int nval);
    private:
        const bool periodic;
        const int nx;
        double * const field;
        double * const comp;
};
