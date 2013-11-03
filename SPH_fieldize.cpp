#include <Python.h>
#include "numpy/arrayobject.h"

/*C...*/
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

/*Compute the SPH weighting for this cell, using the trapezium rule.
 * rr is the smoothing length, r0 is the distance of the cell from the center*/
double compute_sph_cell_weight(double rr, double r0)
{
    if(r0 > rr){
        return 0;
    }
    double total=0;
    double h2 = rr*rr;
    //Do the z integration with the trapezium rule.
    //Evaluate this at some fixed (well-chosen) abcissae
    double zc=0;
    if(rr/2 > r0)
        zc=sqrt(h2/4-r0*r0);
    double zm = sqrt(h2-r0*r0);
    double zz[5]={zc,(3*zc+zm)/4.,(zc+zm)/2.,(zc+3*zm)/2,zm};
    double kern[5];
    double kfac= 8/M_PI/pow(rr,3);
    for(int i=0; i< 5;i++){
        kern[i] = 2*pow(1-sqrt(zz[i]*zz[i]+r0*r0)/rr,3)*kfac;
    }
    for(int i=0; i< 4;i++){
        //Trapezium rule. Factor of 1/2 goes away because we double the integral
        total +=(zz[i+1]-zz[i])*(kern[i+1]+kern[i]);
    }
    if(rr/2 > r0){
        double zz2[9]={0,zc/16.,zc/8.,zc/4.,3*zc/8,zc/2.,5/8.*zc,3*zc/4.,zc};
        double kern2[9];
        for(int i=0; i< 9;i++){
            double R = sqrt(zz2[i]*zz2[i]+r0*r0);
            kern2[i] = kfac*(1-6*(R/rr)*R/rr+6*pow(R/rr,3));
        }
        for(int i=0; i< 8;i++){
            //Trapezium rule. Factor of 1/2 goes away because we double the integral
            total +=(zz2[i+1]-zz2[i])*(kern2[i+1]+kern2[i]);
        }
    }
    return total;
}



#ifndef NO_KAHAN
/*Evaluate one iteration of Kahan Summation: sum is the current value of the field,
 *comp the compensation array, input the value to add this time.*/
inline void KahanSum(double* sum, double* comp, const double input, const int xoff, const int yoff, const int nx)
{
  const int off = nx*xoff+yoff;
  const double yy = input - *(comp+off);
  const double temp = *(sum+off)+yy;     //Alas, sum is big, y small, so low-order digits of y are lost.
  *(comp+off) = (temp - *(sum+off)) -yy; //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
  *(sum+off) = temp;               //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}

#else

inline void KahanSum(double* sum, double* comp, const double input, const int xoff, const int yoff, const int nx)
{
  *(sum+nx*xoff+yoff)+=input;
}
#endif

/**
 Do the hard work interpolating with an SPH kernel particles handed to us from python.
*/
int SPH_interpolate(double * field, double * comp, const int nx, PyArrayObject *pos, PyArrayObject *radii, PyArrayObject *value, PyArrayObject *weights, const npy_int nval, const int periodic)
{
    for(int p=0;p<nval;p++){
        //Temp variables
        float pp[2];
        pp[0]= *(float *)PyArray_GETPTR2(pos,p,1);
        pp[1]= *(float *)PyArray_GETPTR2(pos,p,2);
        float rr= *((float *)PyArray_GETPTR1(radii,p));
        float val= *((float *)PyArray_GETPTR1(value,p));
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
        int upgx = floor(pp[0]+rr);
        int upgy = floor(pp[1]+rr);
        int lowgx = floor(pp[0]-rr);
        int lowgy = floor(pp[1]-rr);
        //Try to save some integrations if this particle is totally in this cell
        if (lowgx==upgx && lowgy==upgy && lowgx >= 0 && lowgy >= 0){
                KahanSum(field, comp, val/weight, lowgx,lowgy,nx);
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
        for(int gy=MAX(lowgy,0);gy<=MIN(upgy,nx-1);gy++)
            for(int gx=MAX(lowgx,0);gx<=MIN(upgx,nx-1);gx++){
                KahanSum(field, comp, val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx,gy,nx);
            }
        //Deal with cells that have wrapped around the edges of the grid
        if (periodic){
            //Wrapping y over
            #pragma omp parallel for
            for(int gy=nx-1;gy<=upgy;gy++){
                //Wrapping only y over
                for(int gx=MAX(lowgx,0);gx<=MIN(upgx,nx-1);gx++){
                    KahanSum(field, comp, val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx,gy-(nx-1),nx);
                }
                //y over, x over
                for(int gx=nx-1;gx<=upgx;gx++){
                    KahanSum(field, comp, val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx-(nx-1),gy-(nx-1),nx);
                }
                //y over, x under
                for(int gx=lowgx;gx<=0;gx++){
                    KahanSum(field, comp, val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx+(nx-1),gy-(nx-1),nx);
                }
            }
            //Wrapping y under
            #pragma omp parallel for
            for(int gy=lowgy;gy<=0;gy++){
                //Only y under
                for(int gx=MAX(lowgx,0);gx<=MIN(upgx,nx-1);gx++){
                    KahanSum(field, comp, val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx,gy+(nx-1),nx);
                }
                //y under, x over
                for(int gx=nx-1;gx<=upgx;gx++){
                    KahanSum(field, comp, val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx-(nx-1),gy+(nx-1),nx);
                }
                //y under, x under
                for(int gx=lowgx;gx<=0;gx++){
                    KahanSum(field, comp, val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx+(nx-1),gy+(nx-1),nx);
                }
            }
            //Finally wrap only x
            #pragma omp parallel for
            for(int gy=MAX(lowgy,0);gy<=MIN(upgy,nx-1);gy++){
                //x over
                for(int gx=nx-1;gx<=upgx;gx++){
                    KahanSum(field, comp, val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx-(nx-1),gy,nx);
                }
                //x under
                for(int gx=lowgx;gx<=0;gx++){
                    KahanSum(field, comp, val*sph_w[gy-lowgy][gx-lowgx]/total/weight,gx+(nx-1),gy,nx);
                }
            }
        }
    }
    return 0;
}
