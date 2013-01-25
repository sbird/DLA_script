"""
Module for computing velocity profile, similar to how they are defined in Rudie et al. 2012 (e.g. see their Figures 5,6)
"""

import numpy as np
import numexpr as ne
import readsubf
import readsubfHDF5
import hdfsim
import h5py
import math
import os.path as path
import cold_gas
import halo_mass_function
import fieldize
import hsml
import scipy.integrate as integ
import scipy.stats
import mpfit

# Take all gas within 3D box around each galaxy. (side of box is 1 physical Mpc; corresponds to (1+z) comoving Mpc)
# For each gas particle in the box, compute the true neutral fraction using Simeon's methods
# Mimic line-of-sight observations by only considering the z-component of the velocity
# Histogram these z-velocities 

    def __init__(self,snap_dir,snapnum,minpart=400,reload_file=False,skip_grid=None,savefile=None):
        self.minpart=minpart
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        #Internal gadget mass unit: 1e10 M_sun/h in g/h
        self.UnitMass_in_g=1.989e43
        #1 M_sun in g
        self.SolarMass_in_g=1.989e33
        #Internal gadget length unit: 1 kpc/h in cm/h
        self.UnitLength_in_cm=3.085678e21
        self.UnitVelocity_in_cm_per_s=1e5
        #Name of savefile
        if savefile == None:
            self.savefile=path.join(self.snap_dir,"snapdir_"+str(self.snapnum).rjust(3,'0'),"halohi_grid.hdf5")
        else:
            self.savefile=savefile
        #For printing

        #regenerate from the raw data
        #Simulation parameters
        f=hdfsim.get_file(snapnum,self.snap_dir,0)
        self.redshift=f["Header"].attrs["Redshift"]
        self.hubble=f["Header"].attrs["HubbleParam"]
        self.box=f["Header"].attrs["BoxSize"]
        self.npart=f["Header"].attrs["NumPart_Total"]+2**32*f["Header"].attrs["NumPart_Total_HighWord"]
        self.omegam=f["Header"].attrs["Omega0"]
        self.omegal=f["Header"].attrs["OmegaLambda"]
        f.close()
        (self.ind,self.sub_mass,self.sub_cofm,self.sub_radii)=self.find_wanted_halos()
        self.nhalo=np.size(self.ind)
        if minpart == -1:
            #global grid
            self.nhalo = 1
            self.sub_radii=np.array([self.box/2.])
        else:
            print "Found ",self.nhalo," halos with > ",minpart,"particles"
        #Set ngrid to be the gravitational softening length
        #self.ngrid=np.array([int(np.ceil(40*self.npart[1]**(1./3)/self.box*2*rr)) for rr in self.sub_radii])
        #self.sub_nHI_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
        #self.sub_gas_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
        #self.set_nHI_grid()


    def set_nHI_grid(self):
        """Set up the grid around each halo where the HI is calculated.
        """
        star=cold_gas.StarFormation(hubble=self.hubble)
        self.once=True
        #Now grid the HI for each halo
        for fnum in xrange(0,500):
            try:
                f=hdfsim.get_file(self.snapnum,self.snap_dir,fnum)
            except IOError:
                break
            print "Starting file ",fnum
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"],dtype=np.float64)
            #Returns neutral density in atoms/cm^3
            irhoH0 = star.get_reproc_rhoHI(bar)
            smooth = hsml.get_smooth_length(bar)
            # gas density in g/cm^3
            irho=np.array(bar["Density"],dtype=np.float64)*(self.UnitMass_in_g/self.UnitLength_in_cm**3)*self.hubble**2
            #Re-use sub_gas_grid to be metallicity
#             irho=np.array(bar["Metallicity"],dtype=np.float64)
            protonmass=1.66053886e-24
            hy_mass = 0.76 # Hydrogen massfrac
            # gas density in hydrogen atoms/cm^3
            irho*=(hy_mass/protonmass)
            f.close()
            #Re-use sub_gas_grid to be molecular hydrogen
#             fH2 = self.get_H2_frac(irhoH0)
#             irhoH2 = irhoH0 * fH2
#             irhoH0 *= (1-fH2)
            #Perform the grid interpolation
            [self.sub_gridize_single_file(ii,ipos,smooth,irho,self.sub_gas_grid,irhoH0,self.sub_nHI_grid) for ii in xrange(0,self.nhalo)]
            #Explicitly delete some things.
            del ipos
            del irhoH0
            del irho
            del smooth
        [np.log1p(grid,grid) for grid in self.sub_gas_grid]
        [np.log1p(grid,grid) for grid in self.sub_nHI_grid]
        #No /= in list comprehensions...  :|
        for i in xrange(0,self.nhalo):
            self.sub_gas_grid[i]/=np.log(10)
            self.sub_nHI_grid[i]/=np.log(10)
        return



