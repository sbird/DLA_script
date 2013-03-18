# vim: set fileencoding=utf-8
"""Module for creating the DLA hydrogen density plots. Can find integrated HI grids around halos (or across the whole box).
   column density functions, cross-sections, etc.

Classes:
    HaloMet - Creates a grid around the halo center with the metal density fraction calculated at each grid cell
"""
import numpy as np
import hdfsim
import os.path as path
import hsml
import h5py

import halohi as hi

class HaloMet(hi.HaloHI):
    """Class to find the integrated metal density around a halo.
    Inherits from HaloMet, and finds grids of metal density in amu / cm^3."""
    def __init__(self,snap_dir,snapnum,elem,minpart=400,reload_file=False,savefile=None):
        self.elem=elem
        self.species = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']
        if savefile==None:
            savefile_s=path.join(snap_dir,"snapdir_"+str(snapnum).rjust(3,'0'),"halomet_grid.hdf5")
        else:
            savefile_s = savefile
        hi.HaloHI.__init__(self,snap_dir,snapnum,minpart=minpart,reload_file=reload_file,savefile=savefile_s)

    def set_nHI_grid(self, gas=False):
        """Set up the grid around each halo where the HI is calculated.
        """
        self.once=True
        nelem = self.species.index(self.elem)
        #Now grid the HI for each halo
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
        files.reverse()
        for ff in files:
            f = h5py.File(ff)
            print "Starting file ",ff
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"])
            #Get HI mass in internal units
            mass=np.array(bar["Masses"])
            #Density in this species
            mass*=np.array(bar["GFM_Metals"][:,nelem],dtype=np.float64)
            smooth = hsml.get_smooth_length(bar)
            [self.sub_gridize_single_file(ii,ipos,smooth,mass,self.sub_nHI_grid) for ii in xrange(0,self.nhalo)]
            f.close()
            #Explicitly delete some things.
            del ipos
            del mass
            del smooth
        #Deal with zeros: 0.1 will not even register for things at 1e17.
        #Also fix the units:
        #we calculated things in internal gadget /cell and we want atoms/cm^2
        #So the conversion is mass/(cm/cell)^2
        for ii in xrange(0,self.nhalo):
            massg=self.UnitMass_in_g/self.hubble*self.hy_mass/self.protonmass
            epsilon=2.*self.sub_radii[ii]/(self.ngrid[ii])*self.UnitLength_in_cm/self.hubble/(1+self.redshift)
            self.sub_nHI_grid[ii]*=(massg/epsilon**2)
            self.sub_nHI_grid[ii]+=0.1
        [np.log10(grid,grid) for grid in self.sub_nHI_grid]
        return
