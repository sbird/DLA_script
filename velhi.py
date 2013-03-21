# -*- coding: utf-8 -*-
"""Derived class for computing velocity phase diagrams from the integrated HI around halos.
   Really not that useful.
"""
import numpy as np
import hdfsim
import os.path as path
import cold_gas
import hsml
from halohi import HaloHI
import matplotlib.pyplot as plt


class VelocityHI(HaloHI):
    """Class for computing velocity diagrams"""
    def __init__(self,snap_dir,snapnum,minpart,reload_file=False,savefile=None):
        if savefile==None:
            savefile_s=path.join(snap_dir,"snapdir_"+str(snapnum).rjust(3,'0'),"velocity_grid.hdf5")
        else:
            savefile_s = savefile
        HaloHI.__init__(self,snap_dir,snapnum,minpart=minpart,reload_file=reload_file,savefile=savefile_s)
        return

    def set_nHI_grid(self):
        """Set up the grid around each halo where the velocity HI is calculated.
        """
        star=cold_gas.RahmatiRT(self.redshift, self.hubble)
        self.once=True
        #This is the real HI grid
        nHI_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
        #Now grid the HI for each halo
        for fnum in xrange(0,500):
            try:
                f=hdfsim.get_file(self.snapnum,self.snap_dir,fnum)
            except IOError:
                break
            print "Starting file ",fnum
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"],dtype=np.float64)
            smooth = hsml.get_smooth_length(bar)
            # Velocity in cm/s
            vel = np.array(bar["Velocities"],dtype=np.float64)*self.UnitVelocity_in_cm_per_s
            #We will weight by neutral mass per cell
            irhoH0 = star.get_reproc_rhoHI(bar)
            irho=np.array(bar["Density"],dtype=np.float64)*(self.UnitMass_in_g/self.UnitLength_in_cm**3)*self.hubble**2
            #HI * Cell Mass, internal units
            mass = np.array(bar["Masses"],dtype=np.float64)*irhoH0/irho
            f.close()
            #Perform the grid interpolation
            #sub_gas_grid is x velocity
            #sub_nHI_grid is y velocity
            [self.sub_gridize_single_file(ii,ipos,smooth,vel[:,1]*mass,self.sub_gas_grid,vel[:,2]*mass,self.sub_nHI_grid,mass) for ii in xrange(0,self.nhalo)]
            #Find the HI density also, so that we can discard
            #velocities in cells that are not DLAs.
            [self.sub_gridize_single_file(ii,ipos,smooth,irhoH0,nHI_grid,np.zeros(np.size(irhoH0)),nHI_grid) for ii in xrange(0,self.nhalo)]
            #Explicitly delete some things.
            del ipos
            del irhoH0
            del irho
            del smooth
            del mass
            del vel
        #No /= in list comprehensions...  :|
        #Average over z
        for i in xrange(0,self.nhalo):
            self.sub_gas_grid[i]/=self.sub_radii[i]
            self.sub_nHI_grid[i]/=self.sub_radii[i]
            ind = np.where(nHI_grid[i] < 10**20.3)
            self.sub_nHI_grid[i][ind] = 0
            self.sub_gas_grid[i][ind] = 0
        return


    def radial_log(self,x,y,cut=1e25):
        """If we have x and y st. x+iy = r e^iθ, find x' and y' s.t. x'+iy' = log(r) e^iθ"""
        r = np.sqrt(x**2+y**2)
        ind = np.where(r > cut)
        sc=np.ones(np.shape(r))
        sc[ind] = cut/r[ind]
        return (x*sc, y*sc)

    def plot_velocity_map(self,num=0,scale=1e32,cut=1e25):
        """Plot the velocity map around a halo"""
        (x,y) = self.radial_log(self.sub_nHI_grid[num],self.sub_gas_grid[num],cut=cut)
        r = np.sqrt(x**2+y**2)
        plt.quiver(x,y,r,scale=scale,scale_units='xy',)

