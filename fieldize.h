#include <Python.h>
#include "numpy/arrayobject.h"
#include <new>
#include <map>

//A class which interpolates data onto a regular grid, projecting
//in the x direction and using SPH interpolation.
class SphInterp
{
    public:
        //field: pointer to the array where interpolated data is stored
        //comp: pointer to temporary memory for Kahan summation. Not used if NO_KAHAN is defined
        //nx: size of the above arrays is nx*nx
        //periodic: should we assume input array is periodic?
        SphInterp(double * field_i, int nx_i, bool periodic_i):
           field(field_i), nx(nx_i), periodic(periodic_i)
        {};
        //pos: array of particle positions
        //radii: particule smoothing lengths
        //value: amount to interpolate to grid
        //weights: weights with which to interpolate
        //nval: size of the above arrays
        int do_work(PyArrayObject *pos, PyArrayObject *radii, PyArrayObject *value, PyArrayObject *weights, const npy_int nval);

        //Direct access to computed data
        double * const field;
        const int nx;
    private:
        const bool periodic;

        inline void KahanSum(const double input, const int xoff, const int yoff)
        {
            *(field+nx*xoff+yoff)+=input;
        }

};

//As above, but interpolation uses Kahan Summation
class KahanSphInterp: public SphInterp
{
    public:
        KahanSphInterp(double * field_i, int nx_i, bool periodic_i):
            SphInterp(field_i, nx_i, periodic_i)
        {
            //Allocate Kahan compensation array, and throw if we can't.
            comp = (double *) calloc(nx*nx,sizeof(double));
            if( !comp )
                throw std::bad_alloc();
        }
        ~KahanSphInterp()
        {
            free(comp);
        };
        /*Evaluate one iteration of Kahan Summation: sum is the current value of the field,
         *comp the compensation array, input the value to add this time.*/
        inline void KahanSum(const double input, const int xoff, const int yoff)
        {
            const int off = nx*xoff+yoff;
            const double yy = input - *(comp+off);
            const double temp = *(field+off)+yy;     //Alas, field is big, y small, so low-order digits of y are lost.
            *(comp+off) = (temp - *(field+off)) -yy; //(t - field) recovers the high-order part of y; subtracting y recovers -(low part of y)
            *(field+off) = temp;               //Algebraically, c should always be zero. Beware eagerly optimising compilers!
        }
    private:
        double * comp;
};


//As above, but discard all interpolation except
//onto a predefined list of array elements
class DiscardingSphInterp: public SphInterp
{
    public:
        DiscardingSphInterp(double * field_i, PyArrayObject * positions, int nx_i, bool periodic_i):
            SphInterp(field_i, nx_i, periodic_i)
        {
            npy_intp nlist = PyArray_SIZE(positions);
            //Build an index of the actual positions of each item we want in the output array.
            for(int i=0; i<nlist;i++)
                index.insert(std::pair<int,int>(*(int *)PyArray_GETPTR1(positions,i),i));
            //Allocate Kahan compensation array, and throw if we can't.
            comp = (double *) calloc(nlist,sizeof(double));
            if( !comp )
                throw std::bad_alloc();
        }

        /*Evaluate one iteration of Kahan Summation: sum is the current value of the field,
         *comp the compensation array, input the value to add this time.*/
        inline void KahanSum(const double input, const int xoff, const int yoff)
        {
            const int off = nx*xoff+yoff;
            std::map<const int, const int>::iterator it = index.find(off);
            if(it != index.end())
            {
                field[it->second]+= input;
                const double yy = input - comp[it->second];
                const double temp = field[it->second]+yy;     //Alas, field is big, y small, so low-order digits of y are lost.
                comp[it->second] = temp - field[it->second] -yy; //(t - field) recovers the high-order part of y; subtracting y recovers -(low part of y)
                field[it->second] = temp;               //Algebraically, c should always be zero. Beware eagerly optimising compilers!
            }
        }

    private:
        std::map<const int, const int> index;
        double * comp;
};
