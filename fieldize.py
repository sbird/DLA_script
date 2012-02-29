"""Methods for interpolating particle lists onto a grid. There are three classic methods:
    ngp - Nearest grid point (point interpolation)
    cic - Cloud in Cell (linear interpolation)
    tsc - Triangular Shaped Cloud (quadratic interpolation)

    Each function takes inputs:
        Values - list of field values to interpolate, centered on the grid center.
        Points - coordinates of the field values
        Field  - grid to add interpolated points onto

    There are also helper functions (convert and convert_centered) to rescale arrays to grid units.
"""

import numpy as np
#Try to import scipy.weave. If we can't, don't worry, we just use the unaccelerated versions
try :
    import scipy.weave
except ImportError :
    scipy=None

def convert(pos, ngrid,box):
    """Rescales coordinates to grid units.
    (0,0) is the lower corner of the grid.
    Inputs:
        pos - coord array to rescale
        ngrid - dimension of grid
        box - Size of the grid in units of pos
    """
    return pos*(ngrid-1)/box

def convert_centered(pos, ngrid,box):
    """Rescales coordinates to grid units.
    (0,0) is the center of the grid
    Inputs:
        pos - coord array to rescale
        ngrid - dimension of grid
        box - Size of the grid in units of pos
    """
    return pos*(ngrid-1.)/float(box)+(ngrid-1.)/2.

def check_input(pos, field):
    """Checks the position and field values for consistency.
    Avoids segfaults in the C code."""
    if np.size(pos) == 0:
        return 0
    dims=np.size(np.shape(field))
    if np.max(pos) > np.shape(field)[0] or np.min(pos) < 0:
        raise ValueError("Positions outside grid")
    if np.shape(pos)[1] < dims:
        raise ValueError("Position array not wide enough for field")
    return 1

def ngp(pos,values,field):
    """Does nearest grid point for a 2D array.
    Inputs:
        Values - list of field values to interpolate
        Points - coordinates of the field values
        Field  - grid to add interpolated points onto

    Points need to be in grid units

    Note: This is implemented in scipy.weave and pure python (in case the weave breaks).
    For O(1e5) points both versions are basically instantaneous.
    For O(1e7) points the sipy.weave version is about 100 times faster.
    """
    if not check_input(pos,field):
        return field
    nx=np.shape(values)[0]
    dims=np.size(np.shape(field))
    # Coordinates of nearest grid point (ngp).
    ind=np.array(np.rint(pos),dtype=np.int)
    #Sum over the 3rd axis here.
    expr="""for(int j=0;j<nx;j++){
            int ind1=ind(j,0);
            int ind2=ind(j,1);
            field(ind1,ind2)+=values(j);
        }
    """
    expr3d="""for(int j=0;j<nx;j++){
            int ind1=ind(j,0);
            int ind2=ind(j,1);
            int ind3=ind(j,2);
            field(ind1,ind2,ind3)+=values(j);
        }
    """
    try:
        if dims==2:
            scipy.weave.inline(expr,['nx','ind','values','field'],type_converters=scipy.weave.converters.blitz)
        elif dims==3:
            scipy.weave.inline(expr3d,['nx','ind','values','field'],type_converters=scipy.weave.converters.blitz)
        else:
            raise ValueError
    except Exception:
        #Fall back on slow python version.
        for j in xrange(0,nx):
            field[tuple(ind[j,0:dims])]+=values[j]
    return field

def cic(pos, values, field):
    """Does nearest grid point for a 2D array.
    Inputs:
        Values - list of field values to interpolate
        Points - coordinates of the field values
        Field  - grid to add interpolated points onto

    Points need to be in coordinates where np.max(points) = np.shape(field)
    """
    if not check_input(pos,field):
        return field
    nx=np.shape(values)[0]
    dims=np.size(np.shape(field))
    raise Exception, "Not Implemented"

def tsc(pos,value,field,totweight=None,periodic=False):
    """ NAME:    TSC

 PURPOSE:
       Interpolate an irregularly sampled field using a Triangular Shaped Cloud

 EXPLANATION:
       This function interpolates an irregularly sampled field to a
       regular grid using Triangular Shaped Cloud (nearest grid point
       gets weight 0.75-dx**2, points before and after nearest grid
       points get weight 0.5*(1.5-dx)**2, where dx is the distance
       from the sample to the grid point in units of the cell size).

 INPUTS:
       pos:  Array of coordinates of field samples, in grid units from 0 to nx
       value: Array of sample weights (field values). For e.g. a
          temperature field this would be the temperature and the
          keyword AVERAGE should be set. For e.g. a density field
          this could be either the particle mass (AVERAGE should
          not be set) or the density (AVERAGE should be set).
       field: Array to interpolate onto of size nx,nx,nx
       totweight: If this is not None, the routine will to it the weights at each
                  grid point. You can then calculate the average later.
       periodic: Set this keyword if you want a periodic grid.
           ie, the first grid point contains samples of both sides of the volume
           If this is not true, weight is not conserved (some falls off the edges)

       Note: Points need to be in grid units: pos = [0,ngrid-1]
       Note 2: If field has fewer dimensions than pos, we sum over the extra dimensions,
               and the final indices are ignored.


 Example of default allocation of nearest grid points: n0=4, *=gridpoint.

     0   1   2   3     Index of gridpoints
     *   *   *   *     Grid points
   |---|---|---|---|   Range allocated to gridpoints ([0.0,1.0> --> 0, etc.)
   0   1   2   3   4   posx


 OUTPUTS:
       Returns particles interpolated to field, and modifies input variable of the same name.

 PROCEDURE:
       Nearest grid point is determined for each sample.
       TSC weights are computed for each sample.
       Samples are interpolated to the grid.
       Grid point values are computed (sum or average of samples).

 EXAMPLE:
       nx=20
       ny=10
       posx=randomu(s,1000)
       posy=randomu(s,1000)
       value=posx**2+posy**2
       field=tsc(value,pos,field,/average)
       surface,field,/lego

 NOTES:
       A standard reference for these interpolation methods is:   R.W. Hockney
       and J.W. Eastwood, Computer Simulations Using Particles (New York:
       McGraw-Hill, 1981).

 MODIFICATION HISTORY:
       Written by Joop Schaye, Feb 1999.
       Check for overflow for large dimensions  P. Riley/W. Landsman Dec. 1999
       Ported to python, cleaned up and drastically shortened using
       these new-fangled "function" thingies by Simeon Bird, Feb. 2012
    """

    # Some error handling.
    if not check_input(pos,field):
        return field

    nval=np.size(value)
    dim=np.shape(field)
    nx = dim[0]
    dim=np.size(dim)

    #-----------------------
    # Calculate TSC weights.
    #-----------------------

    # Coordinates of nearest grid point (ngp).
    ng=np.array(np.rint(pos[:,0:dim]),dtype=np.int)

    # Distance from sample to ngp.
    dng=ng-pos[:,0:dim]

    #Setup two arrays for later:
    # kk is for the indices, and ww is for the weights.
    kk=[np.empty([nval,dim]),np.empty([nval,dim]),np.empty([nval,dim])]
    ww=[np.empty([nval,dim]),np.empty([nval,dim]),np.empty([nval,dim])]
    # Index of ngp.
    kk[1]=ng
    # Weight of ngp.
    ww[1]=0.75-dng**2

    # Point before ngp.
    kk[0]=kk[1]-1  # Index.
    dd=1.0-dng  # Distance to sample.
    ww[0]=0.5*(1.5-dd)**2  # TSC-weight.

    # Point after ngp.
    kk[2]=kk[1]+1  # Index.
    dd=1.0+dng  # Distance to sample.
    ww[2]=0.5*(1.5-dd)**2  # TSC-weight.

    #Take care of the points at the boundaries
    if periodic:
        #If periodic, the nearest grid indices need to wrap around
        #Note python has a sensible remainder operator
        #which always returns > 0 , unlike C
        for axis in xrange(0,2):
            kk[axis]=kk[axis]%nx
    else:
        for axis in xrange(0,2):
            #Find points outside the grid
            ind=np.where((kk[axis] < 0) + (kk[axis] > nx-1))
            #Set the weights of these points to zero
            ww[axis][ind]=0
            #Indices of these points now do not matter, so set to zero also
            kk[axis][ind]=0

    #-----------------------------
    # Interpolate samples to grid.
    #-----------------------------

    # tscweight adds up all tsc weights allocated to a grid point, we need
    # to keep track of this in order to compute the temperature.
    # Note that total(tscweight) is equal to nrsamples and that
    # total(ifield)=n0**3 if sph.plot NE 'sph,temp' (not 1 because we use
    # xpos=posx*n0 --> cube length different from EDFW paper).

    #index[j] -> kk[0][j,0],kk[0][j,2],kk[0][j,3] -> kk[0][j,:]

    extraind=np.zeros(dim-1,dtype=int)
    #Perform y=0, z=0 addition
    tsc_xind(field,value,totweight,kk,ww,extraind)

    if dim > 1:
        #Perform z=0 addition
        for yy in xrange(1,2):
            extraind[0]=yy
            tsc_xind(field,value,totweight,kk,ww,extraind)

    if dim > 2:
        #Perform the rest of the addition
        for zz in xrange(1,2):
            for yy in xrange(0,2):
                extraind[0]=yy
                extraind[1]=zz
                tsc_xind(field,value,totweight,kk,ww,extraind)
    if totweight == None:
        return field
    else:
        return (field,totweight)

def tscadd(field,index,weight,value,totweight):
    """This function is a helper for the tsc and cic routines. It adds
       the weighted value to the field and optionally calculates the total weight.
    Returns nothing, but alters field
    """
    wwval=weight*value
    for j in xrange(0,np.size(wwval)):
        ind=tuple(index[j,:])
        field[ind]+=wwval[j]
        if totweight != None:
            totweight[ind]+=weight[j]
    return

def get_tscweight(ww,ii):
    """Calculates the TSC weight for a particular set of axes.
    ii should be a vector of length dims having values 0,1,2.
    (for CIC a similar thing but ii has values 0,1)
    eg, call as:
        get_tscweight(ww,[0,0,0])
    """
    tscweight=1
    #tscweight = \Pi ww[1]*ww[2]*ww[3]
    for j in xrange(0,np.size(ii)):
        tscweight*=ww[ii[j]][:,j]
    return tscweight

def tsc_xind(field,value,totweight,kk,ww,extraind):
    """Perform the interpolation along the x-axis.
    extraind argument contains the y and z indices, if needed.
    So for a 1d interpolation, extraind=[], for 2d,
    extraind=[y,], for 3d, extraind=[y,z]
    Returns nothing, but alters field
    """
    dims=np.size(extraind)+1
    dim_list=np.zeros(dims,dtype=int)
    dim_list[1:dims]=extraind
    index=np.array(kk[0])
    #Set up the index to have the right kk values depending on the y,z axes
    for i in xrange(1,dims):
        index[:,i]=kk[extraind[i-1]][:,i]
    #Do the addition for each value of x
    for i in xrange(0,2):
        dim_list[0]=i
        tscweight=get_tscweight(ww,dim_list)
        index[:,0]=kk[i][:,0]
        tscadd(field,index,tscweight,value,totweight)
    return

