#include <Python.h>
#include "numpy/arrayobject.h"
#include <algorithm>
#include "fieldize.h"

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


