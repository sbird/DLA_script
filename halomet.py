# vim: set fileencoding=utf-8
"""Module for creating the DLA hydrogen density plots. Can find integrated HI grids around halos (or across the whole box).
   column density functions, cross-sections, etc.

Classes:
    TotalHaloHI - Finds the average HI fraction in a halo
    HaloHI - Creates a grid around the halo center with the HI fraction calculated at each grid cell
"""
import numpy as np
import hdfsim
import os.path as path
import hsml

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

    def set_nHI_grid(self):
        """Function finds the metal density around each halo rather than the HI density"""
        self.once=True
        nelem = self.species.index(self.elem)
        #Now grid the HI for each halo
        for fnum in xrange(0,500):
            try:
                f=hdfsim.get_file(self.snapnum,self.snap_dir,fnum)
            except IOError:
                break
            print "Starting file ",fnum
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"],dtype=np.float64)
            #Returns neutral density in atoms/cm^3 (comoving)
            smooth = hsml.get_smooth_length(bar)
            # gas density in g/cm^3 (comoving)
            irho=np.array(bar["Density"],dtype=np.float64)*(self.UnitMass_in_g/self.UnitLength_in_cm**3)*self.hubble**2
            #Density in this species
            irho*=np.array(bar["GFM_Metals"][:,nelem],dtype=np.float64)
            protonmass=1.66053886e-24
            # gas density in amu /cm^3 (comoving)
            irho/=protonmass
            f.close()
            #Perform the grid interpolation
            [self.sub_gridize_single_file(ii,ipos,smooth,irho,self.sub_nHI_grid) for ii in xrange(0,self.nhalo)]
            #Explicitly delete some things.
            del ipos
            del irho
            del smooth
        [np.log1p(grid,grid) for grid in self.sub_nHI_grid]
        #No /= in list comprehensions...  :|
        for i in xrange(0,self.nhalo):
            self.sub_nHI_grid[i]/=np.log(10)
        return

