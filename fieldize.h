#include <Python.h>
#include "numpy/arrayobject.h"
#include <new>
#include <map>

class Summer
{
    public:
        Summer(double * field_i, int nx_i):
           field(field_i), nx(nx_i)
        {};
/*         virtual void doSum(const double input, const int xoff, const int yoff); */
    protected:
        double * const field;
        const int nx;
};

//A class which interpolates data onto a regular grid, projecting
//in the x direction and using SPH interpolation.
template <class T> class SphInterp
{
    public:
        //field: pointer to the array where interpolated data is stored
        //comp: pointer to temporary memory for Kahan summation. Not used if NO_KAHAN is defined
        //nx: size of the above arrays is nx*nx
        //periodic: should we assume input array is periodic?
        SphInterp(T& sum_i, int nx_i, bool periodic_i):
          sum(sum_i), nx(nx_i), periodic(periodic_i)
        {};
        //pos: array of particle positions
        //radii: particule smoothing lengths
        //value: amount to interpolate to grid
        //weights: weights with which to interpolate
        //nval: size of the above arrays
        int do_work(PyArrayObject *pos, PyArrayObject *radii, PyArrayObject *value, PyArrayObject *weights, const npy_int nval);
/*         { */
/*  */
/*             return 0; */
/*         }; */

    private:
        T& sum;
        const int nx;
        const bool periodic;

};

class SimpleSummer: public Summer
{
    public:
        SimpleSummer(double * field_i, int nx_i):
            Summer(field_i, nx_i)
        {};
        inline void doSum(const double input, const int xoff, const int yoff)
        {
            field[nx*xoff+yoff]+=input;
        }
};

//As above, but interpolation uses Kahan Summation
class KahanSummer: public Summer
{
    public:
        KahanSummer(double * field_i, int nx_i):
            Summer(field_i, nx_i)
        {
            //Allocate Kahan compensation array, and throw if we can't.
            comp = (double *) calloc(nx*nx,sizeof(double));
            if( !comp )
                throw std::bad_alloc();
        }
        ~KahanSummer()
        {
            free(comp);
        };
        /*Evaluate one iteration of Kahan Summation: sum is the current value of the field,
         *comp the compensation array, input the value to add this time.*/
        inline void doSum(const double input, const int xoff, const int yoff)
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
class DiscardingSummer: public Summer
{
    public:
        DiscardingSummer(double * field_i, PyArrayObject * positions, int nx_i):
            Summer(field_i, nx_i)
        {
            npy_intp nlist = PyArray_SIZE(positions);
            //Build an index of the actual positions of each item we want in the output array.
            for(int i=0; i<nlist;i++)
                index.insert(std::pair<int,int>(*(int64_t *)PyArray_GETPTR1(positions,i),i));
            //Allocate Kahan compensation array, and throw if we can't.
            comp = (double *) calloc(nlist,sizeof(double));
            if( !comp )
                throw std::bad_alloc();
        }

        /*Evaluate one iteration of Kahan Summation: sum is the current value of the field,
         *comp the compensation array, input the value to add this time.*/
        inline void doSum(const double input, const int xoff, const int yoff)
        {
            const int off = nx*xoff+yoff;
            std::map<const int, const int>::const_iterator it = index.find(off);
            if(it != index.end())
            {
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


/*Compute the SPH weighting for this cell, using the trapezium rule.
 * rr is the smoothing length, r0 is the distance of the cell from the center*/
double compute_sph_cell_weight(double rr, double r0);

/**
 Do the hard work interpolating with an SPH kernel particles handed to us from python.
*/
template <class T> int SphInterp<T>::do_work(PyArrayObject *pos, PyArrayObject *radii, PyArrayObject *value, PyArrayObject *weights, const npy_int nval)
{
    for(int p=0;p<nval;p++){
        //Temp variables
        float pp[2];
        pp[0]= *(float *)PyArray_GETPTR2(pos,p,1);
        pp[1]= *(float *)PyArray_GETPTR2(pos,p,2);
        const float rr= *((float *)PyArray_GETPTR1(radii,p));
        const float val= *((float *)PyArray_GETPTR1(value,p));
        double weight = 1;
        if (PyArray_DIM(weights,0) == nval){
            weight= *((double *)PyArray_GETPTR1(weights,p));
            //Why do we do this? Because PyArray_DIM(None) == 1.
            //Thus, if we have been passed a single particle,
            //we can set its weight to 0, and cause infinities.
            if (weight == 0)
                weight = 1;
        }
        //Max size of kernel
        const int upgx = floor(pp[0]+rr);
        const int upgy = floor(pp[1]+rr);
        const int lowgx = floor(pp[0]-rr);
        const int lowgy = floor(pp[1]-rr);
        //Try to save some integrations if this particle is totally in this cell
        if (lowgx==upgx && lowgy==upgy && lowgx >= 0 && lowgy >= 0){
                sum.doSum(val/weight, lowgx,lowgy);
                continue;
        }
        /*Array for storing cell weights*/
        double sph_w[upgy-lowgy+1][upgx-lowgx+1];

        /*Total of cell weights*/
        double total=0;
        /* First compute the cell weights.
         * Subsample the cells if the smoothing length is O(1 cell).
         * This is more accurate, and also avoids edge cases where the particle can rest just between a cell.*/
        int nsub=2*((int)(2./rr))+1;
        double subs[nsub];
        /*Spread subsamples evenly across cell*/
        for(int i=0; i < nsub; i++)
            subs[i] = (i+1.)/(1.*nsub+1);
        #pragma omp parallel for reduction(+:total)
        for(int gy=lowgy;gy<=upgy;gy++)
            for(int gx=lowgx;gx<=upgx;gx++){
                sph_w[gy-lowgy][gx-lowgx]=0;
                for(int iy=0; iy< nsub; iy++)
                for(int ix=0; ix< nsub; ix++){
                    double xx = gx-pp[0]+subs[ix];
                    double yy = gy-pp[1]+subs[iy];
                    double r0 = sqrt(xx*xx+yy*yy);
                    sph_w[gy-lowgy][gx-lowgx]+=compute_sph_cell_weight(rr,r0)/nsub/nsub;
                }
                total+=sph_w[gy-lowgy][gx-lowgx];
            }
//         if(total > 1.05)
//           tothigh++;
//         if(total< 0.5)
//           totlow++;
        if(total == 0){
//            fprintf(stderr,"Massless particle detected! rr=%g gy=%d gx=%d nsub = %d pp= %g %g \n",rr,upgy-lowgy,upgx-lowgx, nsub,-pp[0]+lowgx,-pp[1]+lowgy);
            return 1;
        }
        /* Some cells will be only partially in the array: only partially add them.
         * Then add the right fraction to the total array*/

        #pragma omp parallel for
        for(int gy=std::max(lowgy,0);gy<=std::min(upgy,nx-1);gy++)
            for(int gx=std::max(lowgx,0);gx<=std::min(upgx,nx-1);gx++){
                sum.doSum(val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx,gy);
            }
        //Deal with cells that have wrapped around the edges of the grid
        if (periodic){
            //Wrapping y over
            #pragma omp parallel for
            for(int gy=nx-1;gy<=upgy;gy++){
                //Wrapping only y over
                for(int gx=std::max(lowgx,0);gx<=std::min(upgx,nx-1);gx++){
                    sum.doSum(val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx,gy-(nx-1));
                }
                //y over, x over
                for(int gx=nx-1;gx<=upgx;gx++){
                    sum.doSum(val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx-(nx-1),gy-(nx-1));
                }
                //y over, x under
                for(int gx=lowgx;gx<=0;gx++){
                    sum.doSum(val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx+(nx-1),gy-(nx-1));
                }
            }
            //Wrapping y under
            #pragma omp parallel for
            for(int gy=lowgy;gy<=0;gy++){
                //Only y under
                for(int gx=std::max(lowgx,0);gx<=std::min(upgx,nx-1);gx++){
                    sum.doSum(val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx,gy+(nx-1));
                }
                //y under, x over
                for(int gx=nx-1;gx<=upgx;gx++){
                    sum.doSum(val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx-(nx-1),gy+(nx-1));
                }
                //y under, x under
                for(int gx=lowgx;gx<=0;gx++){
                    sum.doSum(val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx+(nx-1),gy+(nx-1));
                }
            }
            //Finally wrap only x
            #pragma omp parallel for
            for(int gy=std::max(lowgy,0);gy<=std::min(upgy,nx-1);gy++){
                //x over
                for(int gx=nx-1;gx<=upgx;gx++){
                    sum.doSum(val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx-(nx-1),gy);
                }
                //x under
                for(int gx=lowgx;gx<=0;gx++){
                    sum.doSum(val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx+(nx-1),gy);
                }
            }
        }
    }
    return 0;
}
