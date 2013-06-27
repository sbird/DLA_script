#include <Python.h>
#include "numpy/arrayobject.h"
#include "fieldize.h"

/*Check whether the passed array has type typename. Returns 1 if it doesn't, 0 if it does.*/
int check_type(PyArrayObject * arr, int npy_typename)
{
  return !PyArray_EquivTypes(PyArray_DESCR(arr), PyArray_DescrFromType(npy_typename));
}

//  int    3*nval arr  nval arr  nval arr   nx*nx arr  int     nval arr  (or 0)
//['nval','pos',     'radii',    'value',   'field',   'nx',    'weights']
PyObject * Py_SPH_Fieldize(PyObject *self, PyObject *args)
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
    double * comp = (double *) calloc(nx*nx,sizeof(double));
    if( !comp || !field ){
      PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for field arrays.\n");
      return NULL;
    }
    //Do the work
    ret = SPH_interpolate(field, comp, nx, pos, radii, value, weights, nval, periodic);
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

static PyMethodDef __fieldize[] = {
  {"_SPH_Fieldize", Py_SPH_Fieldize, METH_VARARGS,
   "Interpolate particles onto a grid using SPH interpolation."
   "    Arguments: pos, radii, value, weights, periodic=T/F, nx"
   "    "},
  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
init_fieldize_priv(void)
{
  Py_InitModule("_fieldize_priv", __fieldize);
  import_array();
}
