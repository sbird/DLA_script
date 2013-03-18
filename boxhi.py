# -*- coding: utf-8 -*-
"""Derived class for computing the integrated HI across the whole box.
"""
import numpy as np
import os.path as path
import fieldize
from halohi import HaloHI



class BoxHI(HaloHI):
    """Class for calculating a large grid encompassing the whole simulation.
    Stores a big grid projecting the neutral hydrogen along the line of sight for the whole box.

    Parameters:
        dir - Simulation directory
        snapnum - Number of simulation
        reload_file - Ignore saved files if true
        self.sub_nHI_grid is a neutral hydrogen grid, in log(N_HI / cm^-2) units.
        self.sub_mass is a list of halo masses
        self.sub_cofm is a list of halo positions"""
    def __init__(self,snap_dir,snapnum,reload_file=False,savefile=None,gas=False):
        if savefile==None:
            savefile_s=path.join(snap_dir,"snapdir_"+str(snapnum).rjust(3,'0'),"boxhi_grid.hdf5")
        else:
            savefile_s = savefile
        self.nhalo = 1
        HaloHI.__init__(self,snap_dir,snapnum,minpart=-1,reload_file=reload_file,savefile=savefile_s,gas=gas)
        #global grid
        self.sub_pos=np.array([self.box/2., self.box/2.,self.box/2.])
        return

    def sub_gridize_single_file(self,ii,ipos,ismooth,mHI,sub_nHI_grid,weights=None):
        """Helper function for sub_nHI_grid
            that puts data arrays loaded from a particular file onto the grid.
            Arguments:
                pos - Position array
                rho - Density array to be interpolated
                smooth - Smoothing lengths
                sub_grid - Grid to add the interpolated data to
        """
        #coords in grid units
        coords=fieldize.convert(ipos,self.ngrid[0],self.box)
        #NH0
        # Convert each particle's density to column density by multiplying by the smoothing length once (in physical cm)!
        cellspkpc=(self.ngrid[ii]/self.box)
        if self.once:
            avgsmth=np.mean(ismooth)
            print ii," Av. smoothing length is ",avgsmth," kpc/h ",avgsmth*cellspkpc, "grid cells min: ",np.min(ismooth)*cellspkpc
            self.once=False
        #Convert smoothing lengths to grid coordinates.
        ismooth*=cellspkpc

        fieldize.sph_str(coords,mHI,sub_nHI_grid[0],ismooth,weights=weights, periodic=True)
        return

