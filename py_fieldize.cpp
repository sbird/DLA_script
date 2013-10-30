#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "fieldize.h"
#include <map>

/*Check whether the passed array has type typename. Returns 1 if it doesn't, 0 if it does.*/
int check_type(PyArrayObject * arr, int npy_typename)
{
  return !PyArray_EquivTypes(PyArray_DESCR(arr), PyArray_DescrFromType(npy_typename));
}

//  int    3*nval arr  nval arr  nval arr   nx*nx arr  int     nval arr  (or 0)
//['nval','pos',     'radii',    'value',   'field',   'nx',    'weights']
extern "C" PyObject * Py_SPH_Fieldize(PyObject *self, PyObject *args)
{
    PyArrayObject *pos, *radii, *value, *weights;
    int periodic, nx, ret;
    if(!PyArg_ParseTuple(args, "O!O!O!O!ii",&PyArray_Type, &pos, &PyArray_Type, &radii, &PyArray_Type, &value, &PyArray_Type, &weights,&periodic, &nx) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use pos, radii, value, weights periodic=False, nx\n");
        return NULL;
    }
    if(check_type(pos, NPY_FLOAT) || check_type(radii, NPY_FLOAT) || check_type(value, NPY_FLOAT) || check_type(weights, NPY_DOUBLE))
    {
          PyErr_SetString(PyExc_AttributeError, "Input arrays do not have appropriate type: pos, radii and value need float32, weights float64.\n");
          return NULL;
    }
    const npy_intp nval = PyArray_DIM(radii,0);
    if(nval != PyArray_DIM(value,0) || nval != PyArray_DIM(pos,0))
    {
      PyErr_SetString(PyExc_ValueError, "pos, radii and value should have the same length.\n");
      return NULL;
    }
//     int totlow=0, tothigh=0;
    //Field for the output.
    npy_intp size[2]={nx,nx};
    PyArrayObject * pyfield = (PyArrayObject *) PyArray_SimpleNew(2, size, NPY_DOUBLE);
    PyArray_FILLWBYTE(pyfield, 0);
    double * field = (double *) PyArray_DATA(pyfield);
    //Copy of field array to store compensated bits for Kahan summation
#ifndef NO_KAHAN
    double * comp = (double *) calloc(nx*nx,sizeof(double));
    if( !comp || !field ){
#else
    double * comp = NULL;
    if( !field ){
#endif
      PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for field arrays.\n");
      return NULL;
    }
    //Do the work
    ret = SPH_interpolate(field, comp, nx, pos, radii, value, weights, nval, periodic);
    if (comp)
        free(comp);

    if( ret == 1 ){
      PyErr_SetString(PyExc_ValueError, "Massless particle detected!");
      return NULL;
    }
    //printf("Total high: %d total low: %d (%ld)\n",tothigh, totlow,nval);
    PyObject * for_return = Py_BuildValue("O",pyfield);
    Py_DECREF(pyfield);
    return for_return;
}


extern "C" PyObject * Py_find_halo_kernel(PyObject *self, PyObject *args)
{
    PyArrayObject *sub_cofm, *sub_mass, *sub_radii, *xcoords, *ycoords, *zcoords, *dla_cross;
    double box;
    if(!PyArg_ParseTuple(args, "dO!O!O!O!O!O!O!",&box, &PyArray_Type, &sub_cofm, &PyArray_Type, &sub_mass, &PyArray_Type, &sub_radii, &PyArray_Type, &xcoords, &PyArray_Type, &ycoords,&PyArray_Type, &zcoords, &PyArray_Type, &dla_cross) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use box, sub_cofm, sub_mass, sub_radii, xcells, ycells, zcells, dla_cross\n");
        return NULL;
    }
    if(check_type(sub_cofm, NPY_DOUBLE) || check_type(sub_mass, NPY_DOUBLE) || check_type(sub_radii, NPY_DOUBLE) 
            || check_type(xcoords, NPY_DOUBLE) || check_type(ycoords, NPY_DOUBLE) || check_type(zcoords, NPY_DOUBLE) || check_type(dla_cross, NPY_DOUBLE))
    {
          PyErr_SetString(PyExc_AttributeError, "Input arrays do not have appropriate type: all should be double.\n");
          return NULL;
    }

    const npy_intp ncells = PyArray_SIZE(xcoords);
    long int field_dlas = 0;

    #pragma omp parallel for
    for (npy_intp i=0; i< ncells; i++)
    {
        const double xcoord =  (*(double *) PyArray_GETPTR1(xcoords,i));
        const double ycoord =  (*(double *) PyArray_GETPTR1(ycoords,i));
        const double zcoord =  (*(double *) PyArray_GETPTR1(zcoords,i));
        // Largest halo where the particle is within r_vir.
        int nearest_halo=-1;
        for (int j=0; j < PyArray_DIM(sub_cofm,0); j++)
        {
            double xpos = fabs(*(double *) PyArray_GETPTR2(sub_cofm,j,0) - xcoord);
            double ypos = fabs(*(double *) PyArray_GETPTR2(sub_cofm,j,1) - ycoord);
            double zpos = fabs(*(double *) PyArray_GETPTR2(sub_cofm,j,2) - zcoord);
            //Periodic wrapping
            if (xpos > box/2.)
                xpos = box-xpos;
            if (ypos > box/2.)
                ypos = box-ypos;
            if (zpos > box/2.)
                zpos = box-zpos;

            //Distance
            double dd = xpos*xpos + ypos*ypos + zpos*zpos;
            //Is it close?
            double rvir = pow(*(double *) PyArray_GETPTR1(sub_radii,j), 2);
            if (dd < rvir) {
                if (nearest_halo > 0)
                  printf("This should never happen!: part %ld halo %d\n",i,j);
                //Is it a larger mass than the current halo?
                if (nearest_halo < 0 || (*(double *) PyArray_GETPTR1(sub_mass,j) > *(double *) PyArray_GETPTR1(sub_mass,nearest_halo)) ) {
                    nearest_halo = j;
                }
            }
        }
        if (nearest_halo >= 0){
            #pragma omp critical (_dla_cross_)
            {
                *(double *) PyArray_GETPTR1(dla_cross,nearest_halo) += 1.;
            }
        }
        else{
            #pragma omp atomic
            field_dlas++;
        }
    }

    return Py_BuildValue("l",field_dlas);
}

extern "C" PyObject * Py_calc_distance_kernel(PyObject *self, PyObject *args)
{
    PyArrayObject *pos, *mass, *xpos, *ypos, *zpos, *hidist, *himasses;
    double slabsz, gridsz;
    if(!PyArg_ParseTuple(args, "O!O!ddO!O!O!O!O!",&PyArray_Type, &pos, &PyArray_Type, &mass, &slabsz, &gridsz, &PyArray_Type, &xpos, &PyArray_Type, &ypos,&PyArray_Type, &zpos, &PyArray_Type, &hidist, &PyArray_Type, &himasses) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use pos, mass,slabsz, gridsz, xpos, ypos,zpos,hidist, himasses\n");
        return NULL;
    }
    if(check_type(pos, NPY_FLOAT) || check_type(mass, NPY_FLOAT)
            || check_type(xpos, NPY_DOUBLE) || check_type(ypos, NPY_DOUBLE) || check_type(zpos, NPY_DOUBLE)
            || check_type(hidist, NPY_DOUBLE) || check_type(himasses, NPY_DOUBLE))
    {
          PyErr_SetString(PyExc_AttributeError, "Input arrays do not have appropriate type: all should be double except mass and pos.\n");
          return NULL;
    }

    const npy_intp npart = PyArray_SIZE(mass);
    const npy_intp ndlas = PyArray_SIZE(xpos);

    //Copy the data into a map, which automatically sorts it. I don't really need to do this, 
    //but it is easier than writing my own iterator class for the python array.
    std::map<float,int> zvalarr;
    for (int i=0; i< npart; i++)
    {
        zvalarr[*(float *) PyArray_GETPTR2(pos,i,2)] = i;
    }
    #pragma omp parallel for
    // Largest halo where the particle is within r_vir.
    for (npy_intp j=0; j < ndlas; j++)
    {
        double xxdla = *(double *) PyArray_GETPTR1(xpos, j);
        double yydla = *(double *) PyArray_GETPTR1(ypos, j);
        double zzdla = *(double *) PyArray_GETPTR1(zpos, j);
        std::map<float,int>::iterator lower = zvalarr.lower_bound(zzdla-gridsz/2.);
        std::map<float,int>::iterator upper = zvalarr.upper_bound(zzdla+gridsz/2.);
        for (std::map<float,int>::iterator it=lower; it != upper; ++it)
        {
            const double xxdist = fabs(*(float *) PyArray_GETPTR2(pos,it->second,0)-xxdla);
            const double yydist = fabs(*(float *) PyArray_GETPTR2(pos,it->second,1)-yydla);
            const double zzdist = fabs(*(float *) PyArray_GETPTR2(pos,it->second,2)-zzdla);
            //Is it in this cell?
            if (xxdist < slabsz/2. && yydist < gridsz/2. && zzdist < gridsz/2.)
            {
                #pragma omp critical (_himass_)
                {
                    const float mmass = (*(float *) PyArray_GETPTR1(mass,it->second));
                    *(double *) PyArray_GETPTR1(hidist,j) += mmass*( *(float *) PyArray_GETPTR2(pos,it->second,0));
                    *(double *) PyArray_GETPTR1(himasses,j) += mmass;
                }
                break;
            }
        }
    }
    int i=0;
    return Py_BuildValue("i",&i);
}



static PyMethodDef __fieldize[] = {
  {"_SPH_Fieldize", Py_SPH_Fieldize, METH_VARARGS,
   "Interpolate particles onto a grid using SPH interpolation."
   "    Arguments: pos, radii, value, weights, periodic=T/F, nx"
   "    "},
  {"_find_halo_kernel", Py_find_halo_kernel, METH_VARARGS,
   "Kernel for populating a field containing the mass of the nearest halo to each point"
   "    Arguments: sub_cofm, sub_mass, sub_radii, xcells, ycells, zcells (output from np.where), dla_cross[nn]"
   "    "},
  {"_calc_distance_kernel", Py_calc_distance_kernel, METH_VARARGS,
   "Kernel for finding the HI weighted distance"
   "    Arguments: pos, mass,slabsz, gridsz, xpos, ypos,zpos,hidist, himasses"
   "    "},

  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
init_fieldize_priv(void)
{
  Py_InitModule("_fieldize_priv", __fieldize);
  import_array();
}
