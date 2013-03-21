#include <Python.h>
#include "numpy/arrayobject.h"

#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

/*Note we need some contiguous memory space after the actual data in field. *The real input data has size
 *dims*dims*dims
 *The output has size dims*dims*(dims/2+1) *complex* values
 * So need dims*dims*dims+2 float space.
 * Also the field needs to be stored carefully to make the 
 * extra space be in the right place. */

/**Little macro to work the storage order of the FFT.*/
#define KVAL(n) ((n)<=dims/2 ? (n) : ((n)-dims))

/*This is the power spectrum frm a 2d field*/
int powerspectrum(int dims, fftw_complex *outfield, int nrbins, double *power, int *count, double *keffs)
{
	const int dims2=dims*dims;
	/*How many bins per unit interval in k?*/
	const int binsperunit=nrbins/(floor(sqrt(2)*abs((dims+1.0)/2.0)+1));
	/*Half the bin width*/
	const double bwth=1.0/(2.0*binsperunit);
	/* Now we compute the powerspectrum in each direction.
	 * FFTW is unnormalised, so we need to scale by the length of the array
	 * (we do this later). */
	memset(power, 0,nrbins*sizeof(double));
	memset(count, 0,nrbins*sizeof(int));
	for(int i=0; i< nrbins/2; i++){
		/* bin center is k=i+a.
		 * a is bin width/2 (bwth)
		 * Effective bin center, k_eff is 
		 * expected modes per interval.
		 * k_eff = int(k d^2k) k-a -> k+a
		 *            / int(d^2k) k-a -> k+a
		 * k_eff is k+ a^2/(3k) */
		double k=i*2.0*bwth+bwth;
		keffs[i]=k+bwth*bwth/3/k;
	}
	/*After this point, the number of modes is decreasing, so use the inverse of the above.*/
	for(int i=nrbins/2; i< nrbins; i++){
		double k=i*2.0*bwth+bwth;
		keffs[i]=k-bwth*bwth/3/k;
	}
	#pragma omp parallel 
	{
		double powerpriv[nrbins];
		int countpriv[nrbins];
		memset(powerpriv, 0,nrbins*sizeof(double));
		memset(countpriv, 0,nrbins*sizeof(int));
		/* Want P(k)= F(k).re*F(k).re+F(k).im*F(k).im
		 * Use the symmetry of the real fourier transform to half the final dimension.*/
		#pragma omp for schedule(static, 128) nowait
		for(int j=0; j<dims; j++){
			int indy=j*(dims/2+1);
			/* The k=0 and N/2 mode need special treatment here, 
			 * as they alone are not doubled.*/
			/*Do k=0 mode.*/
			int index=indy;
			double kk=abs(KVAL(j));
			int psindex=floor(binsperunit*kk);
			powerpriv[psindex]+=(pow(outfield[index][0],2)+pow(outfield[index][1],2));
			countpriv[psindex]++;
			/*Now do the k=N/2 mode*/
			index=indy+dims/2;
			kk=sqrt(KVAL(j)*KVAL(j)+KVAL(dims/2)*KVAL(dims/2));
			psindex=floor(binsperunit*kk);
			powerpriv[psindex]+=(pow(outfield[index][0],2)+pow(outfield[index][1],2));
			countpriv[psindex]++;
			/*Now do the rest. Because of the symmetry, each mode counts twice.*/
			for(int k=1; k<dims/2; k++){
				index=indy+k;
				kk=sqrt(KVAL(j)*KVAL(j)+KVAL(k)*KVAL(k));
				psindex=floor(binsperunit*kk);
				powerpriv[psindex]+=2*(pow(outfield[index][0],2)+pow(outfield[index][1],2));
				countpriv[psindex]+=2;
			}
		}
		#pragma omp critical
		{
			for(int i=0; i< nrbins;i++){
				power[i]+=powerpriv[i];
				count[i]+=countpriv[i];
			}
		}
	}
	for(int i=0; i< nrbins;i++){
		if(count[i]){
			/* I do the division twice to avoid any overflow.*/
			power[i]/=dims2;
			power[i]/=dims2;
			power[i]/=count[i];
		}
	}
	return 0;
}


/*Wrap power spectrum calculation into python*/
PyObject * Py_powerspectrum_2d(PyObject *self, PyObject *args)
{
  PyArrayObject *field_p;
  int nrbins;
  if(!PyArg_ParseTuple(args, "O!i",&PyArray_Type, &field_p,&nrbins) )
        return NULL;
  //Assume square
  const npy_intp field_dims = PyArray_DIM(field_p,0);
  npy_intp nrbins_p = nrbins;
  //Memory for the field
  /* Allocate extra memory to do an out-of-place transform.*/
  fftw_complex * outfield=(fftw_complex *) fftw_malloc(field_dims*(field_dims/2+1)*sizeof(fftw_complex));
  double * field = (double *) PyArray_DATA(PyArray_GETCONTIGUOUS(field_p));
  //Allocate memory for output
  double * power=(double *) malloc(nrbins*sizeof(double));
  int * count=(int *) malloc(nrbins*sizeof(int));
  double * keffs=(double *) malloc(nrbins*sizeof(double));
  if(!outfield || !power || !count || !keffs){
  	fprintf(stderr,"Error allocating memory.\n");
        return NULL;
  }

  //Set up FFTW
  if(!fftw_init_threads()){
  		  fprintf(stderr,"Error initialising fftw threads\n");
  		  return NULL;
  }

  //Do the FFT
  fftw_plan_with_nthreads(omp_get_num_procs());
  fftw_plan pl=fftw_plan_dft_r2c_2d(field_dims,field_dims,&field[0],outfield, FFTW_ESTIMATE);
  fftw_execute(pl);
  fftw_destroy_plan(pl);
  /*Now make a power spectrum for each particle type*/
  powerspectrum(field_dims,outfield,nrbins, power,count,keffs);
  fftw_free(outfield);
  
  //Copy data back into python arrays
  PyArrayObject *keffs_p = (PyArrayObject *) PyArray_SimpleNew(1,&nrbins_p,NPY_DOUBLE);
  PyArrayObject *power_p = (PyArrayObject *) PyArray_SimpleNew(1,&nrbins_p,NPY_DOUBLE);
  PyArrayObject *count_p = (PyArrayObject *) PyArray_SimpleNew(1,&nrbins_p,NPY_DOUBLE);
  for(int nn=0; nn< nrbins; nn++){
            *(double *)PyArray_GETPTR1(keffs_p,nn)=keffs[nn];
	    *(double *)PyArray_GETPTR1(power_p,nn)=power[nn];
	    *(double *)PyArray_GETPTR1(count_p,nn)=count[nn];
  }
  //Free memory
  free(power);
  free(count);
  free(keffs);

  return Py_BuildValue("OOO", keffs_p, power_p,count_p);
}


static PyMethodDef __powerspec[] = {
  {"powerspectrum_2d", Py_powerspectrum_2d, METH_VARARGS,
   "Find the power spectrum of a 2D density field.\n "
   "Arguments: field. \n"
   "Output: (keff (box units), power (fourier units), count.)  Will be binned evenly in k"
   "    "},
  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
init_power_priv(void)
{
  Py_InitModule("_power_priv", __powerspec);
  import_array();
}
