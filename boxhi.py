# -*- coding: utf-8 -*-
"""Derived class for computing the integrated HI across the whole box.
"""
import numpy as np
import os.path as path
import fieldize
import numexpr as ne
from halohi import HaloHI



class BoxHI(HaloHI):
    """Class for calculating a large grid encompassing the whole simulation.
    Stores a big grid projecting the neutral hydrogen along the line of sight for the whole box.

    Parameters:
        dir - Simulation directory
        snapnum - Number of simulation
        reload_file - Ignore saved files if true
        nslice - number of slices in the z direction to divide the box into.
    """
    def __init__(self,snap_dir,snapnum,nslice=1,reload_file=False,savefile=None,gas=False):
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        self.set_units()
        if savefile==None:
            self.savefile = path.join(snap_dir,"snapdir_"+str(snapnum).rjust(3,'0'),"boxhi_grid.hdf5")
        else:
            self.savefile = savefile
        self.sub_mass = 10.**12*np.ones(nslice)
        self.nhalo = nslice
        try:
            if reload_file:
                raise KeyError("reloading")
            #First try to load from a file
            self.load_savefile(self.savefile)
        except (IOError,KeyError):
            #Otherwise regenerate from the raw data
            self.load_header()
            #global grid in slices
            self.sub_cofm=0.5*np.ones([nslice,3])
            self.sub_cofm[:,0]=(np.arange(0,nslice)+0.5)/(1.*nslice)*self.box
            self.sub_radii=self.box/2.*np.ones(nslice)
            #Grid size double softening length
            self.ngrid=np.array([int(np.ceil(40*self.npart[1]**(1./3)/self.box*2*rr)) for rr in self.sub_radii])/2.
            self.sub_nHI_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
            self.set_nHI_grid(gas)
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
        #At the moment any particle which is located in the slice is wholly in the slice:
        #no periodicity or smoothing.
        #Gather all particles in this slice
        #Is the avg smoothing length is ~100 kpc and the slice is ~2.5 Mpc wide, this will be a small effect.
        jpos = self.sub_cofm[ii,0]
        jjpos = ipos[:,0]
        grid_radius = self.box/self.nhalo/2.
        indj = np.where(ne.evaluate("abs(jjpos-jpos) < grid_radius"))

        if np.size(indj) == 0:
            return

        ipos = ipos[indj]
        # Update smooth and rho arrays as well:
        ismooth = ismooth[indj]
        mHI = mHI[indj]

        #coords in grid units
        coords=fieldize.convert(ipos,self.ngrid[0],self.box)
        # Convert each particle's density to column density by multiplying by the smoothing length once (in physical cm)!
        cellspkpc=(self.ngrid[0]/self.box)
        if self.once:
            avgsmth=np.mean(ismooth)
            print ii," Av. smoothing length is ",avgsmth," kpc/h ",avgsmth*cellspkpc, "grid cells min: ",np.min(ismooth)*cellspkpc
            self.once=False
        #Convert smoothing lengths to grid coordinates.
        ismooth*=cellspkpc

        fieldize.sph_str(coords,mHI,sub_nHI_grid[ii],ismooth,weights=weights, periodic=True)
        return

    def absorption_distance(self):
        """Compute X(z), the absorption distance per sightline (eq. 9 of Nagamine et al 2003)
        in dimensionless units, accounting for slicing the box."""
        #h * 100 km/s/Mpc in h/s
        h100=3.2407789e-18
        # in cm/s
        light=2.9979e10
        #Units: h/s   s/cm                        kpc/h      cm/kpc
        return h100/light*(1+self.redshift)**2*(self.box/self.nhalo)*self.UnitLength_in_cm


    def omega_DLA(self, thresh=20.3):
        """Compute Omega_DLA, the sum of the mass in DLAs, divided by the critical density.
            Ω_DLA = m_p * avg. column density / (1+z)^2 / length of column / rho_c
            Note: If we want the neutral gas density rather than the neutral hydrogen density, divide by 0.76,
            the hydrogen mass fraction.
        """
        #Average column density of HI in atoms cm^-2 (physical)
        if thresh > 0:
            HImass = np.array([np.sum(10**grid[np.where(grid > thresh)])/np.size(grid) for grid in self.sub_nHI_grid])
            HImass = np.mean(HImass)
        else:
            HImass = np.mean(10**self.sub_nHI_grid)
        #Avg. Column density of HI in g cm^-2 (comoving)
        HImass = self.protonmass * HImass/(1+self.redshift)**2
        #Length of column in comoving cm
        length = (self.box*self.UnitLength_in_cm/self.hubble/self.nhalo)
        #Avg density in g/cm^3 (comoving) divided by critical density in g/cm^3
        omega_DLA=HImass/length/self.rho_crit()
        return omega_DLA
