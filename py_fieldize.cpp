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
    if( !field ){
      PyErr_SetString(PyExc_MemoryError, "Passed a null field array!.\n");
      return NULL;
    }
    //Do the work
    try {
#ifdef NO_KAHAN
        SimpleSummer sum(field, nx);
        SphInterp<SimpleSummer> worker(sum, nx, periodic);
#else
        KahanSummer sum(field, nx);
        SphInterp<KahanSummer> worker(sum, nx, periodic);
#endif
        ret = worker.do_work(pos, radii, value, weights, nval);
    }
    catch (std::bad_alloc &) {
      PyErr_SetString(PyExc_MemoryError, "Could not allocate Kahan compensation array!\n");
      return NULL;
    }
    if( ret == 1 ){
      PyErr_SetString(PyExc_ValueError, "Massless particle detected!");
      return NULL;
    }
    //printf("Total high: %d total low: %d (%ld)\n",tothigh, totlow,nval);
    PyObject * for_return = Py_BuildValue("O",pyfield);
    Py_DECREF(pyfield);
    return for_return;
}

extern "C" PyObject * Py_Discard_SPH_Fieldize(PyObject *self, PyObject *args)
{
    PyArrayObject *pos, *radii, *value, *weights, *field_list;
    int periodic, nx, ret;
    if(!PyArg_ParseTuple(args, "O!O!O!O!O!ii",&PyArray_Type, &field_list, &PyArray_Type, &pos, &PyArray_Type, &radii, &PyArray_Type, &value, &PyArray_Type, &weights,&periodic, &nx) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use field_list, pos, radii, value, weights periodic=False, nx\n");
        return NULL;
    }
    if(check_type(field_list, NPY_INT64) || check_type(pos, NPY_FLOAT) || check_type(radii, NPY_FLOAT) || check_type(value, NPY_FLOAT) || check_type(weights, NPY_DOUBLE))
    {
          PyErr_SetString(PyExc_AttributeError, "Input arrays do not have appropriate type: field_list needs int32, pos, radii and value need float32, weights float64.\n");
          return NULL;
    }
    const npy_intp nval = PyArray_DIM(radii,0);
    if(nval != PyArray_DIM(value,0) || nval != PyArray_DIM(pos,0))
    {
      PyErr_SetString(PyExc_ValueError, "pos, radii and value should have the same length.\n");
      return NULL;
    }
    //Field for the output.
    npy_intp nlist = PyArray_SIZE(field_list);
    PyArrayObject * pyfield = (PyArrayObject *) PyArray_SimpleNew(1, &nlist, NPY_DOUBLE);
    PyArray_FILLWBYTE(pyfield, 0);
    double * field = (double *) PyArray_DATA(pyfield);
    //Copy of field array to store compensated bits for Kahan summation
    if( !field ){
      PyErr_SetString(PyExc_MemoryError, "Passed a null field array!.\n");
      return NULL;
    }
    //Do the work
    try {
        DiscardingSummer sum(field, field_list, nx);
        SphInterp<DiscardingSummer> worker(sum, nx, periodic);
        ret = worker.do_work(pos, radii, value, weights, nval);
    }
    catch (std::bad_alloc &) {
      PyErr_SetString(PyExc_MemoryError, "Could not allocate Kahan compensation array!\n");
      return NULL;
    }
    if( ret == 1 ){
      PyErr_SetString(PyExc_ValueError, "Massless particle detected!");
      return NULL;
    }
    //printf("Total high: %d total low: %d (%ld)\n",tothigh, totlow,nval);
    PyObject * for_return = Py_BuildValue("O",pyfield);
    Py_DECREF(pyfield);
    return for_return;
}

//Test whether a particle with position (xcoord, ycoord, zcoord)
//is within the virial radius of halo j.
inline bool is_halo_close(const int j, const double xcoord, const double ycoord, const double zcoord, PyArrayObject * sub_cofm, PyArrayObject * sub_radii, const double box)
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
            const double dd = xpos*xpos + ypos*ypos + zpos*zpos;
            //Is it close?
            const double rvir = pow(*(double *) PyArray_GETPTR1(sub_radii,j), 2);
            //We will only be within the virial radius for one halo
            if (dd <= rvir) {
                return true;
            }
            else
                return false;
}

extern "C" PyObject * Py_find_halo_kernel(PyObject *self, PyObject *args)
{
    PyArrayObject *sub_cofm, *sub_radii, *sub_mass, *xcoords, *ycoords, *zcoords, *dla_cross, *assigned_halo, *subsub_pos, *subsub_radii, *subsub_index;
    double box;
    if(!PyArg_ParseTuple(args, "dO!O!O!O!O!O!O!O!O!O!O!",&box, &PyArray_Type, &sub_cofm, &PyArray_Type, &sub_radii, &PyArray_Type, &sub_mass, &PyArray_Type, &subsub_pos, &PyArray_Type, &subsub_radii, &PyArray_Type, &subsub_index, &PyArray_Type, &xcoords, &PyArray_Type, &ycoords,&PyArray_Type, &zcoords, &PyArray_Type, &dla_cross, &PyArray_Type, &assigned_halo) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use box, halo_cofm, halo_radii, halo_mass, sub_pos, sub_radii, sub_index, xcells, ycells, zcells, dla_cross, assigned_halo\n");
        return NULL;
    }
    if(check_type(sub_mass, NPY_DOUBLE) || check_type(sub_cofm, NPY_DOUBLE) || check_type(sub_radii, NPY_DOUBLE) 
            || check_type(subsub_radii, NPY_DOUBLE) || check_type(subsub_pos, NPY_DOUBLE) || check_type(subsub_index, NPY_INT32)
            || check_type(xcoords, NPY_DOUBLE) || check_type(ycoords, NPY_DOUBLE) || check_type(zcoords, NPY_DOUBLE) || check_type(dla_cross, NPY_DOUBLE)
            ||  check_type(assigned_halo, NPY_INT32))
    {
          PyErr_SetString(PyExc_AttributeError, "Input arrays do not have appropriate type: all should be double except assigned_halo and sub index.\n");
          return NULL;
    }

    if(PyArray_NDIM(xcoords) != 1 || PyArray_NDIM(ycoords) != 1 || PyArray_NDIM(zcoords) != 1)
    {
          PyErr_SetString(PyExc_AttributeError, "Input DLA coordinates are not 1D\n");
          return NULL;
    }

    if(PyArray_NDIM(sub_cofm) != 2 || PyArray_NDIM(subsub_pos) != 2 )
    {
          PyErr_SetString(PyExc_AttributeError, "Halo positions coordinates are not 3D\n");
          return NULL;
    }

    const npy_intp ncells = PyArray_SIZE(xcoords);
    const npy_intp nhalo = PyArray_DIM(sub_cofm,0);
    const npy_intp nsubhalo = PyArray_SIZE(subsub_radii);

    long int field_dlas = 0;
    //Store index in a map as the easiest way of sorting it
    std::map<const double, const int> sort_mass;
    //Insert - the mass into the map, so that the largest halo comes first.
    for (int i=0; i< nhalo; ++i){
        sort_mass.insert(std::pair<const double, const int>(-1*(*(double *) PyArray_GETPTR1(sub_mass,i)),i));
    }

    //Store index in a map as the easiest way of sorting it
    std::multimap<const int32_t, const int> sort_sub_index;
    //Insert - the index into the map, so that the subhalos of the largest halo comes first.
    for (int i=0; i< nsubhalo; ++i){
        sort_sub_index.insert(std::pair<const int32_t, const int>(*(int *) PyArray_GETPTR1(subsub_index,i),i));
    }
    #pragma omp parallel for
    for (npy_intp i=0; i< ncells; i++)
    {
        const double xcoord =  (*(double *) PyArray_GETPTR1(xcoords,i));
        const double ycoord =  (*(double *) PyArray_GETPTR1(ycoords,i));
        const double zcoord =  (*(double *) PyArray_GETPTR1(zcoords,i));

        // Largest halo where the particle is within r_vir.
        int nearest_halo=-1;
        for (std::map<const double,const int>::const_iterator it = sort_mass.begin(); it != sort_mass.end(); ++it)
        {
            if (is_halo_close(it->second, xcoord, ycoord, zcoord, sub_cofm, sub_radii, box)) {
                nearest_halo = it->second;
                break;
            }
        }
        //If no halo found, loop over subhalos.
        if (nearest_halo <= 0){
          for (std::multimap<const int32_t,const int>::const_iterator it = sort_sub_index.begin(); it != sort_sub_index.end(); ++it){
            //If close to a subhalo, assign to the parent halo.
            if (is_halo_close(it->second, xcoord, ycoord, zcoord, subsub_pos, subsub_radii, box)) {
                nearest_halo = it->first;
                break;
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
        *(int32_t *) PyArray_GETPTR1(assigned_halo,i) = nearest_halo;
    }

    return Py_BuildValue("l",field_dlas);
}

static PyMethodDef __fieldize[] = {
  {"_SPH_Fieldize", Py_SPH_Fieldize, METH_VARARGS,
   "Interpolate particles onto a grid using SPH interpolation."
   "    Arguments: pos, radii, value, weights, periodic=T/F, nx"
   "    "},
  {"_find_halo_kernel", Py_find_halo_kernel, METH_VARARGS,
   "Kernel for populating a field containing the mass of the nearest halo to each point"
   "    Arguments: halo_cofm, halo_radii, halo_mass, sub_pos, sub_radii, sub_index, xcells, ycells, zcells (output from np.where), dla_cross[nn], assigned_halo"
   "    "},
  {"_Discard_SPH_Fieldize", Py_Discard_SPH_Fieldize, METH_VARARGS,
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
