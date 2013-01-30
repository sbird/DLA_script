#include <Python.h>
#include "numpy/arrayobject.h"

#define IL 128

void empty_cache(double * value, int * id1, int * id2, PyArrayObject * field, int il)
{
    #pragma omp critical
    {
    for(int i=0; i< il; i++)
         *((double *) PyArray_GETPTR2(field,id1[i],id2[i]))+=value[i];
    }
}

void add_to_data_array(double * result, int * id1, int * id2, PyArrayObject * field, int * il, int gx, int gy, double value)
{
            if(*il > IL -1){
                empty_cache(result, id1, id2, field, *il);
                *il=0;
            }
            result[*il] = value;
            id1[*il] = gx;
            id2[*il] = gy;
            (*il)++;
}

//  int    3*nval arr  nval arr  nval arr   nx*nx arr  int     nval arr  (or 0)
//['nval','pos',     'radii',    'value',   'field',   'nx',    'weights']
PyObject * Py_SPH_Fieldize(PyObject *self, PyObject *args)
{
    PyArrayObject *pos, *radii, *value, *field, *weights;
    if(!PyArg_ParseTuple(args, "O!O!O!O!O!",&PyArray_Type, &pos, &PyArray_Type, &radii, &PyArray_Type, &value, &PyArray_Type, &field, &PyArray_Type, &weights) )
        return NULL;
    const npy_intp nval = PyArray_DIM(radii,0);
    const npy_intp nx = PyArray_DIM(field,0);
    int il = 0;
/*     #pragma omp parallel for firstprivate(il) */
    for(int p=0;p<nval;p++){
        //Thread-local cache
        double result[IL];
        int id1[IL], id2[IL];
        //Temp variables
        double pp[2];
        pp[0]= *(double *)PyArray_GETPTR2(pos,p,1);
        pp[1]= *(double *)PyArray_GETPTR2(pos,p,2);
        double rr= *((double *)PyArray_GETPTR1(radii,p));
        double val= *((double *)PyArray_GETPTR1(value,p));
        double weight = 1;
        if (PyArray_DIM(weights,0) == nval){
            weight= *((double *)PyArray_GETPTR1(weights,p));
            //Why do we do this? Because PyArray_DIM(None) == 1.
            //Thus, if we have been passed a single particle,
            //we can set its weight to 0, and cause infinities.
            if (weight == 0)
                weight = 1;
        }
        //99% of the kernel is inside 0.85 of the smoothing length.
        //Neglect the rest.
        int upgx = floor(pp[0]+0.85*rr);
        int upgy = floor(pp[1]+0.85*rr);
        int lowgx = floor(pp[0]-0.85*rr);
        int lowgy = floor(pp[1]-0.85*rr);
        //Try to save some integrations if this particle is totally in this cell
        if (lowgx==upgx && lowgy==upgy && lowgx >= 0 && lowgy >= 0){
                *((double *) PyArray_GETPTR2(field,lowgx,lowgy))+=val/weight;
/*             add_to_data_array(result, id1, id2, field, &il, lowgx, lowgy, val/weight); */
        }
        else {
            //Deal with the edges
            if(upgx > nx-1)
                upgx=nx-1;
            if(upgy > nx-1)
                upgy=nx-1;
            if(lowgx < 0)
                lowgx=0;
            if(lowgy < 0)
                lowgy=0;
            for(int gy=lowgy;gy<=upgy;gy++)
                for(int gx=lowgx;gx<=upgx;gx++){
                    double total=0;
                    double xx = gx-pp[0]+0.5;
                    double yy = gy-pp[1]+0.5;
                    double r0 = sqrt(xx*xx+yy*yy);
                    if(r0 > rr)
                        continue;
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
/*                     add_to_data_array(result, id1, id2, field, &il, gx, gy, val*total/weight); */
                    *((double *) PyArray_GETPTR2(field,gx,gy))+=val*total/weight;
/*                     field(gx,gy)+=val*total/weight; */
                }
        }
        /*Empty on final iteration*/
/*         if(p == nval-1) */
/*             empty_cache(result, id1, id2, field,il); */
        }
	return Py_BuildValue("i",nval);
}

static PyMethodDef __fieldize[] = {
  {"_SPH_Fieldize", Py_SPH_Fieldize, METH_VARARGS,
   "Interpolate particles onto a grid using SPH interpolation."
   "    Arguments: nbins, pos, vel, mass, u, nh0, ne, h, axis array, xx, yy, zz"
   "    "},
  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
init_fieldize_priv(void)
{
  Py_InitModule("_fieldize_priv", __fieldize);
  import_array();
}
