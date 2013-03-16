# -*- coding: utf-8 -*-
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
import math
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

def cic(pos, value, field,totweight=None,periodic=False):
    """Does Cloud-in-Cell for a 2D array.
    Inputs:
        Values - list of field values to interpolate
        Points - coordinates of the field values
        Field  - grid to add interpolated points onto

    Points need to be in coordinates where np.max(points) = np.shape(field)
    """
    # Some error handling.
    if not check_input(pos,field):
        return field

    nval=np.size(value)
    dim=np.shape(field)
    nx = dim[0]
    dim=np.size(dim)

    #-----------------------
    # Calculate CIC weights.
    #-----------------------

    # Coordinates of nearest grid point (ngp).
    ng=np.array(np.rint(pos[:,0:dim]),dtype=np.int)

    # Distance from sample to ngp.
    dng=ng-pos[:,0:dim]

    #Setup two arrays for later:
    # kk is for the indices, and ww is for the weights.
    kk=np.empty([2,nval,dim])
    ww=np.empty([2,nval,dim])
    # Index of ngp.
    kk[1]=ng
    # Weight of ngp.
    ww[1]=0.5+np.abs(dng)

    # Point before ngp.
    kk[0]=kk[1]-1  # Index.
    ww[0]=0.5-np.abs(dng)

    #Take care of the points at the boundaries
    tscedge(kk,ww,nx,periodic)

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
        extraind[0]=1
        tsc_xind(field,value,totweight,kk,ww,extraind)

    if dim > 2:
        extraind[1]=1
        #Perform the rest of the addition
        for yy in xrange(0,2):
            extraind[0]=yy
            tsc_xind(field,value,totweight,kk,ww,extraind)
    if totweight == None:
        return field
    else:
        return (field,totweight)


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
    kk=np.empty([3,nval,dim])
    ww=np.empty([3,nval,dim])
    # Index of ngp.
    kk[1,:,:]=ng
    # Weight of ngp.
    ww[1,:,:]=0.75-dng**2

    # Point before ngp.
    kk[0,:,:]=kk[1,:,:]-1  # Index.
    dd=1.0-dng  # Distance to sample.
    ww[0]=0.5*(1.5-dd)**2  # TSC-weight.

    # Point after ngp.
    kk[2,:,:]=kk[1,:,:]+1  # Index.
    dd=1.0+dng  # Distance to sample.
    ww[2]=0.5*(1.5-dd)**2  # TSC-weight.

    #Take care of the points at the boundaries
    tscedge(kk,ww,nx,periodic)

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
        for yy in xrange(1,3):
            extraind[0]=yy
            tsc_xind(field,value,totweight,kk,ww,extraind)

    if dim > 2:
        #Perform the rest of the addition
        for zz in xrange(1,3):
            for yy in xrange(0,3):
                extraind[0]=yy
                extraind[1]=zz
                tsc_xind(field,value,totweight,kk,ww,extraind)
    if totweight == None:
        return field
    else:
        return (field,totweight)

def cic_str(pos,value,field,in_radii,periodic=False):
    """This is exactly the same as the cic() routine, above, except
       that instead of each particle being stretched over one grid point,
       it is stretched over a cubic region with some radius.

       Field must be 2d
       Extra arguments:
            radii - Array of particle radii in grid units.
    """
    # Some error handling.
    if not check_input(pos,field):
        return field

    nval=np.size(value)
    dim=np.shape(field)
    nx = dim[0]
    dim=np.size(dim)
    if dim != 2:
        raise ValueError("Non 2D grid not supported!")
    #Use a grid cell radius of 2/3 (4 \pi /3 )**(1/3) s
    #This means that l^3 = cell volume for AREPO (so it should be more or less exact)
    #and is close to the l = 0.5 (4\pi/3)**(1/3) s
    #cic interpolation that Nagamine, Springel & Hernquist used
    #to approximate their SPH smoothing
    corr=2./3.*(4*math.pi/3.)**0.3333333333
    radii=np.array(corr*in_radii)
    #If the smoothing length is below a single grid cell,
    #stretch it.
    ind = np.where(radii < 0.5)
    radii[ind]=0.5
    #Weight of each cell
    weight = value/(2*radii)**dim
    #Upper and lower bounds
    up = pos[:,1:dim+1]+np.repeat(np.transpose([radii,]),dim,axis=1)
    low = pos[:,1:dim+1]-np.repeat(np.transpose([radii,]),dim,axis=1)
    #Upper and lower grid cells to add to
    upg = np.array(np.floor(up),dtype=int)
    lowg = np.array(np.floor(low),dtype=int)
    #Deal with the edges
    if periodic:
        raise ValueError("Periodic grid not supported")
    else:
        ind=np.where(up > nx-1)
        up[ind] = nx
        upg[ind]=nx-1
        ind=np.where(low < 0)
        low[ind]=0
        lowg[ind]=0

    expr="""for(int p=0;p<nval;p++){
            //Temp variables
            double wght = weight(p);
            int ilx=lowg(p,0);
            int ily=lowg(p,1);
            int iux=upg(p,0);
            int iuy=upg(p,1);
            double lx=low(p,0);
            double ly=low(p,1);
            double ux=up(p,0);
            double uy=up(p,1);
            //Deal with corner values
            field(ilx,ily)+=(ilx+1-lx)*(ily+1-ly)*wght;
            field(iux,ily)+=(ux-iux)*(ily+1-ly)*wght;
            field(ilx,iuy)+=(ilx+1-lx)*(uy-iuy)*wght;
            field(iux,iuy)+=(ux-iux)*(uy-iuy)*wght;
            //Edges in y
            for(int gx=ilx+1;gx<iux;gx++){
                field(gx,ily)+=(ily+1-ly)*wght;
                field(gx,iuy)+=(uy-iuy)*wght;
            }
            //Central region
            for(int gy=ily+1;gy< iuy;gy++){
                //Edges.
                field(ilx,gy)+=(ilx+1-lx)*wght;
                field(iux,gy)+=(ux-iux)*wght;
                //x-values
                for(int gx=ilx+1;gx<iux;gx++){
                    field(gx,gy)+=wght;
                }
            }
        }
    """
    try:
        scipy.weave.inline(expr,['nval','upg','lowg','field','up','low','weight'],type_converters=scipy.weave.converters.blitz)
    except Exception:
        for p in xrange(0,nval):
            #Deal with corner values
            field[lowg[p,0],lowg[p,1]]+=(lowg[p,0]+1-low[p,0])*(lowg[p,1]+1-low[p,1])*weight[p]
            field[upg[p,0],lowg[p,1]]+=(up[p,0]-upg[p,0])*(lowg[p,1]+1-low[p,1])*weight[p]
            field[lowg[p,0],upg[p,1]]+=(lowg[p,0]+1-low[p,0])*(up[p,1]-upg[p,1])*weight[p]
            field[upg[p,0], upg[p,1]]+=(up[p,0]-upg[p,0])*(up[p,1]-upg[p,1])*weight[p]
            #Edges in y
            for gx in xrange(lowg[p,0]+1,upg[p,0]):
                field[gx,lowg[p,1]]+=(lowg[p,1]+1-low[p,1])*weight[p]
                field[gx,upg[p,1]]+=(up[p,1]-upg[p,1])*weight[p]
            #Central region
            for gy in xrange(lowg[p,1]+1,upg[p,1]):
                #Edges in x
                field[lowg[p,0],gy]+=(lowg[p,0]+1-low[p,0])*weight[p]
                field[upg[p,0],gy]+=(up[p,0]-upg[p,0])*weight[p]
                #x-values
                for gx in xrange(lowg[p,0]+1,upg[p,0]):
                    field[gx,gy]+=weight[p]
    return field

from _fieldize_priv import _SPH_Fieldize

def sph_str(pos,value,field,radii,weights=None,periodic=False):
    """Interpolate a particle onto the grid using an SPH kernel.
       This is similar to the cic_str() routine, but spherical.

       Field must be 2d
       Extra arguments:
            radii - Array of particle radii in grid units.
            weights - Weights to divide each contribution by.
    """
    # Some error handling.
    if not check_input(pos,field):
        return field

    dim=np.shape(field)
    dim=np.size(dim)
    if dim != 2:
        raise ValueError("Non 2D grid not supported!")
    if weights == None:
        weights = np.array([0.])
    #Cast some array types
    if pos.dtype != np.float32:
       pos = np.array(pos, dtype=float32)
    if radii.dtype != np.float32:
       radii = np.array(radii, dtype=float32)
    if value.dtype != np.float32:
        value = np.array(value, dtype=float32)
    nval = _SPH_Fieldize(pos, radii, value, field, weights,periodic)
    if nval < np.size(value):
        raise ValueError("Something went wrong with interpolation")
    return field

import scipy.integrate as integ

def integrate_sph_kernel(h,gx,gy):
    """Compute the integrated sph kernel for a particle with
       smoothing length h, at position pos, for a grid-cell at gg"""
    #Fast method; use the value at the grid cell.
    #Bad if h < grid cell radius
    r0 = np.sqrt((gx+0.5)**2+(gy+0.5)**2)
    if r0 > h:
        return 0
    h2 = h*h
    #Do the z integration with the trapezium rule.
    #Evaluate this at some fixed (well-chosen) abcissae
    zc=0
    if h/2 > r0:
        zc=np.sqrt(h2/4-r0**2)
    zm = np.sqrt(h2-r0**2)
    zz=np.array([zc,(3*zc+zm)/4.,(zc+zm)/2.,(zc+3*zm)/2,zm])
    kern = sph_kern2(np.sqrt(zz**2+r0**2),h)
    total= 2*integ.simps(kern,zz)
    if h/2 > r0:
        zz=np.array([0,zc/8.,zc/4.,3*zc/8,zc/2.,5/8.*zc,3*zc/4.,zc])
        kern = sph_kern1(np.sqrt(zz**2+r0**2),h)
        total+= 2*integ.simps(kern,zz)
    return total

def do_slow_sph_integral(h,gx,gy):
    """Evaluate the very slow triple integral to find kernel contribution. Only do it when we must."""
    #z limits are -h - > h, for simplicity.
    #x and y limits are grid cells
    (weight,err)=integ.tplquad(sph_cart_wrap,-h,h,lambda x: gx,lambda x: gx+1,lambda x,y: gy,lambda x,y:gy+1,args=(h,),epsabs=5e-3)
    return weight

def sph_cart_wrap(z,y,x,h):
    """Cartesian wrapper around sph_kernel"""
    r = np.sqrt(x**2+y**2+z**2)
    return sph_kernel(r,h)

def sph_kern1(r,h):
    """SPH kernel for 0 < r < h/2"""
    return 8/math.pi/h**3*(1-6*(r/h)**2+6*(r/h)**3)

def sph_kern2(r,h):
    """SPH kernel for h/2 < r < h"""
    return 2*(1-r/h)**3*8/math.pi/h**3

def sph_kernel(r,h):
    """Evaluates the sph kernel used in gadget."""
    if r > h:
        return 0
    elif r > h/2:
        return 2*(1-r/h)**3*8/math.pi/h**3
    else:
        return 8/math.pi/h**3*(1-6*(r/h)**2+6*(r/h)**3)

def tscedge(kk,ww,ngrid,periodic):
    """This function takes care of the points at the grid boundaries,
       either by wrapping them around the grid (the Julie Andrews sense)
       or by throwing them over the side (the Al Pacino sense).
       Arguments are:
           kk - the grid indices
           ww - the grid weights
           nx - the number of grid points
           periodic - Julie or Al?
    """
    if periodic:
        #If periodic, the nearest grid indices need to wrap around
        #Note python has a sensible remainder operator
        #which always returns > 0 , unlike C
        kk=kk%ngrid
    else:
        #Find points outside the grid
        ind=np.where(np.logical_or((kk < 0),(kk > ngrid-1)))
        #Set the weights of these points to zero
        ww[ind]=0
        #Indices of these points now do not matter, so set to zero also
        kk[ind]=0


def tscadd(field,index,weight,value,totweight):
    """This function is a helper for the tsc and cic routines. It adds
       the weighted value to the field and optionally calculates the total weight.
    Returns nothing, but alters field
    """
    nx=np.size(value)
    dims=np.size(np.shape(field))
    total=totweight !=None
    #Faster C version of this function: this is getting a little out of hand.
    expr="""for(int j=0;j<nx;j++){
        int ind1=index(j,0);
        int ind2=index(j,1);
        """
    if dims == 3:
        expr+="""int ind3=index(j,2);
                 field(ind1,ind2,ind3)+=weight(j)*value(j);
                 """
        if total:
            expr+=" totweight(ind1,ind2,ind3) +=weight(j);"
    if dims == 2:
        expr+="""field(ind1,ind2)+=weight(j)*value(j);
              """
        if total:
            expr+=" totweight(ind1,ind2) +=weight(j);"
    expr+="}"
    try:
        if dims==2 or dims == 3:
            if total:
                scipy.weave.inline(expr,['nx','index','value','field','weight','totweight'],type_converters=scipy.weave.converters.blitz)
            else:
                scipy.weave.inline(expr,['nx','index','value','field','weight'],type_converters=scipy.weave.converters.blitz)
        else:
            raise ValueError
    except Exception:
        wwval=weight*value
        for j in xrange(0,nx):
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
    tscweight=1.
    #tscweight = \Pi ww[1]*ww[2]*ww[3]
    for j in xrange(0,np.size(ii)):
        tscweight*=ww[ii[j],:,j]
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
    index=kk[0]
    #Set up the index to have the right kk values depending on the y,z axes
    for i in xrange(1,dims):
        index[:,i]=kk[extraind[i-1],:,i]
    #Do the addition for each value of x
    for i in xrange(0,np.shape(kk)[0]):
        dim_list[0]=i
        tscweight=get_tscweight(ww,dim_list)
        index[:,0]=kk[i,:,0]
        tscadd(field,index,tscweight,value,totweight)
    return

