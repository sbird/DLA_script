import numpy as np
import scipy.weave

def ngp(pos,values,grid):
        """Does nearest grid point. 
        
        ==Parameters==

        Values is a list of field values
        Points are the coordinates of the field values
        Grid is the grid to interpolate onto

        Points need to be in coordinates where max(points) = np.shape(grid)"""
        # Coordinates of nearest grid point (ngp).
        ind=np.array(np.rint(pos),dtype=np.int)
        nx=np.shape(values)[0]
        #Slow python version
        #for j in xrange(0,nx):
        #        grid[ind[j,:]]+=values[j]
        #Sum over the 3rd axis here.
        expr="""for(int j=0;j<nx;j++){
                        int ind1=ind(j,0);
                        int ind2=ind(j,1);
                        grid(ind1,ind2)+=values(j);
                }
        """
        scipy.weave.inline(expr,['nx','ind','values','grid'],type_converters=scipy.weave.converters.blitz)
        return grid

def tsc(value,pos,field,average=True,wraparound=False,no_message=True,isolated=True):
        """ NAME:    TSC

 PURPOSE:
       Interpolate an irregularly sampled field using a Triangular Shaped Cloud

 EXPLANATION:
       This function interpolates an irregularly sampled field to a
       regular grid using Triangular Shaped Cloud (nearest grid point
       gets weight 0.75-dx**2, points before and after nearest grid
       points get weight 0.5*(1.5-dx)**2, where dx is the distance
       from the sample to the grid point in units of the cell size).

 CATEGORY:
       Mathematical functions, Interpolation

 INPUTS:
       VALUE: Array of sample weights (field values). For e.g. a
              temperature field this would be the temperature and the
              keyword AVERAGE should be set. For e.g. a density field
              this could be either the particle mass (AVERAGE should
              not be set) or the density (AVERAGE should be set).
       POS:  Array of coordinates of field samples, in grid units from 0 to nx
       FIELD: Array to interpolate onto of size nx,nx,nx
       

 KEYWORD PARAMETERS:
       AVERAGE:    Set this keyword if the nodes contain field samples
                   (e.g. a temperature field). The value at each grid
                   point will then be the weighted average of all the
                   samples allocated to it. If this keyword is not
                   set, the value at each grid point will be the
                   weighted sum of all the nodes allocated to it
                   (e.g. for a density field from a distribution of
                   particles). (D=0). 
       WRAPAROUND: Set this keyword if you want the first grid point
                   to contain samples of both sides of the volume
                   (see below).
       ISOLATED:   Set this keyword if the data is isolated, i.e. not
                   periodic. In that case total `mass' is not conserved.
                   This keyword cannot be used in combination with the
                   keyword WRAPAROUND.
       NO_MESSAGE: Suppress informational messages.

 Example of default allocation of nearest grid points: n0=4, *=gridpoint.

     0   1   2   3     Index of gridpoints
     *   *   *   *     Grid points
   |---|---|---|---|   Range allocated to gridpoints ([0.0,1.0> --> 0, etc.)
   0   1   2   3   4   posx

 Example of ngp allocation for WRAPAROUND: n0=4, *=gridpoint.

   0   1   2   3         Index of gridpoints
   *   *   *   *         Grid points
 |---|---|---|---|--     Range allocated to gridpoints ([0.5,1.5> --> 1, etc.)
   0   1   2   3   4=0   posx


 OUTPUTS:
       Prints that a TSC interpolation is being performed of x
       samples to y grid points, unless NO_MESSAGE is set.

 RESTRICTIONS:
       Field data is assumed to be periodic with the sampled volume
       the basic cell, unless ISOLATED is set.
       All input arrays must have the same dimensions.
       Postition coordinates should be in `index units' of the
       desired grid: POSX=[0,NX>, etc.
       Keywords ISOLATED and WRAPAROUND cannot both be set.

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
       Use csc.pro or ngp.pro for lower order interpolation schemes.    A 
       standard reference for these interpolation methods is:   R.W. Hockney 
       and J.W. Eastwood, Computer Simulations Using Particles (New York: 
       McGraw-Hill, 1981).

 MODIFICATION HISTORY:
       Written by Joop Schaye, Feb 1999.
       Check for overflow for large dimensions  P. Riley/W. Landsman Dec. 1999
       Ported to python by Simeon Bird 2012"""
        nrsamples=np.size(value)
        dim=np.shape(pos)
        nx = dim[0]
        if np.size(dim) > 1:
                dim = dim[1]
        else:
                dim = 1
        nxny=nx**2
        
        
        #---------------------
        # Some error handling.
        #---------------------
        if nrsamples != nx:
                raise ValueError,"Weights must have same length as values"
        
        if isolated and wraparound:
                raise ValueError,'Keywords ISOLATED and WRAPAROUND cannot both be set!'
        
        if not no_message:
                print 'Interpolating ',nrsamples,' samples to ',nx**3,' grid points using TSC...'
        
        
        #-----------------------
        # Calculate TSC weights.
        #-----------------------
        
        # Coordinates of nearest grid point (ngp).
        if wraparound:
                ng=np.trunc(pos+0.5)
        else:
                ng=np.trunc(pos)+0.5
        
        # Distance from sample to ngp.
        dng=ng-pos
        
        # Index of ngp.
        if wraparound:
                k2=ng
        else:
                k2=ng-0.5
        # Weight of ngp.
        w2=0.75-dng**2
        
        # Point before ngp.
        k1=k2-1  # Index.
        dd=1.0-dng  # Distance to sample.
        w1=0.5*(1.5-dd)**2  # TSC-weight.
        
        # Point after ngp.
        k3=k2+1  # Index.
        dd=1.0+dng  # Distance to sample.
        w3=0.5*(1.5-dd)**2  # TSC-weight.
        
        # Periodic boundary conditions.
        bad=np.where(k2 == 0)
        if np.size(bad) > 0:       # Otherwise kx1=-1.
                kx1[bad]=nx-1
                if isolated:
                        wx1[bad]=0.
        bad=np.where(k2 == nx-1)
        if np.size(bad) > 0:       # Otherwise kx3=nx.
                kx3[bad]=0
                if isolated:
                        wx3[bad]=0.
        if wraparound:
                bad=np.where(k2 == nx)
                if np.size(bad) > 0:
                        k2[bad]=0
                        k3[bad]=1
        del bad  # Free memory.
        
        #-----------------------------
        # Interpolate samples to grid.
        #-----------------------------
        
        if average:
                tottscweight=np.zeros(np.shape(field))
        
        # tscweight adds up all tsc weights allocated to a grid point, we need
        # to keep track of this in order to compute the temperature.
        # Note that total(tscweight) is equal to nrsamples and that
        # total(ifield)=n0**3 if sph.plot NE 'sph,temp' (not 1 because we use
        # xpos=posx*n0 --> cube length different from EDFW paper).
        
        #index[j] -> k1[j,0],k1[j,2],k1[j,3] -> k1[j,:]

        tscweight=w1[:,0]
        if dim > 1:
                tscweight*=w1[:,1]
                if dim > 2:
                        tscweight*=w1[:,2]
        index=np.array(k1)
        for j in xrange(0,nrsamples):
                field[index[j,:]]+=tscweight[j]*value[j]
                if average:
                        tottscweight[k1[j,:]]+=tscweight[j]
        #tscweight=w2[:,0]*w1[:,1]*w1[:,2]
        tscweight*=w2[:,0]/w1[:,0]
        index[:,0]=k2[:,0]
        for j in xrange(0,nrsamples):
                field[index[j,:]]+=tscweight[j]*value[j]
                if average:
                        tottscweight[index[j,:]]+=tscweight[j]
        index[:,0]=k3[:,0]
        tscweight*=w3[:,0]/w2[:,0]
        for j in xrange(0,nrsamples):
                field[index[j,:]]+=tscweight[j]*value[j]
                if average:
                        tottscweight[index[j,:]]+=tscweight[j]

        if dim > 1:
                index=np.array(k1)
                index[:,1]=k2[:,1]
                #Reset
                tscweight*=w1[:,0]/w3[:,0]
                tscweight*=w2[:,1]/w1[:,1]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[k1[j,:]]+=tscweight[j]
                #tscweight=w2[:,0]*w1[:,1]*w1[:,2]
                tscweight*=w2[:,0]/w1[:,0]
                index[:,0]=k2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,0]=k3[:,0]
                tscweight*=w3[:,0]/w2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index=np.array(k1)
                index[:,1]=k3[:,1]
                #Reset
                tscweight*=w1[:,0]/w3[:,0]
                tscweight*=w3[:,1]/w2[:,1]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[k1[j,:]]+=tscweight[j]
                #tscweight=w2[:,0]*w1[:,1]*w1[:,2]
                tscweight*=w2[:,0]/w1[:,0]
                index[:,0]=k2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,0]=k3[:,0]
                tscweight*=w3[:,0]/w2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]

        if dim > 2:
                tscweight=w1[:,0]*w1[:,1]*w2[:,2]
                index=np.array(k1)
                index[:,2]=k2[:,2]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[k1[j,:]]+=tscweight[j]
                #tscweight=w2[:,0]*w1[:,1]*w1[:,2]
                tscweight*=w2[:,0]/w1[:,0]
                index[:,0]=k2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,0]=k3[:,0]
                tscweight*=w3[:,0]/w2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,1]=k2[:,1]
                #Reset
                tscweight*=w1[:,0]/w3[:,0]
                tscweight*=w2[:,1]/w1[:,1]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[k1[j,:]]+=tscweight[j]
                #tscweight=w2[:,0]*w1[:,1]*w1[:,2]
                tscweight*=w2[:,0]/w1[:,0]
                index[:,0]=k2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,0]=k3[:,0]
                tscweight*=w3[:,0]/w2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index=np.array(k1)
                index[:,1]=k3[:,1]
                #Reset
                tscweight*=w1[:,0]/w3[:,0]
                tscweight*=w3[:,1]/w2[:,1]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[k1[j,:]]+=tscweight[j]
                #tscweight=w2[:,0]*w1[:,1]*w1[:,2]
                tscweight*=w2[:,0]/w1[:,0]
                index[:,0]=k2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,0]=k3[:,0]
                tscweight*=w3[:,0]/w2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                tscweight=w1[:,0]*w1[:,1]*w3[:,2]
                index=np.array(k1)
                index[:,2]=k3[:,2]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[k1[j,:]]+=tscweight[j]
                #tscweight=w2[:,0]*w1[:,1]*w1[:,2]
                tscweight*=w2[:,0]/w1[:,0]
                index[:,0]=k2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,0]=k3[:,0]
                tscweight*=w3[:,0]/w2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,1]=k2[:,1]
                #Reset
                tscweight*=w1[:,0]/w3[:,0]
                tscweight*=w2[:,1]/w1[:,1]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[k1[j,:]]+=tscweight[j]
                #tscweight=w2[:,0]*w1[:,1]*w1[:,2]
                tscweight*=w2[:,0]/w1[:,0]
                index[:,0]=k2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,0]=k3[:,0]
                tscweight*=w3[:,0]/w2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index=np.array(k1)
                index[:,1]=k3[:,1]
                #Reset
                tscweight*=w1[:,0]/w3[:,0]
                tscweight*=w3[:,1]/w2[:,1]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[k1[j,:]]+=tscweight[j]
                #tscweight=w2[:,0]*w1[:,1]*w1[:,2]
                tscweight*=w2[:,0]/w1[:,0]
                index[:,0]=k2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]
                index[:,0]=k3[:,0]
                tscweight*=w3[:,0]/w2[:,0]
                for j in xrange(0,nrsamples):
                        field[index[j,:]]+=tscweight[j]*value[j]
                        if average:
                                tottscweight[index[j,:]]+=tscweight[j]

        #--------------------------
        # Compute weighted average.
        #--------------------------
        
        if average:
                good=np.where(tottscweight != 0)
                field[good]=field[good]/tottscweight[good]
        
        return field
