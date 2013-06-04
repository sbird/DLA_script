# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8
"""Module for creating the DLA hydrogen density plots. Can find integrated HI grids around halos (or across the whole box).
   column density functions, cross-sections, etc.

Classes:
    HaloMet - Creates a grid around the halo center with the metal density fraction calculated at each grid cell
"""
import numpy as np
import hdfsim
import convert_cloudy
import os.path as path
import hsml
import h5py
import numexpr as ne
import halohi as hi
import boxhi as bi
import fieldize

class HaloMet(hi.HaloHI):
    """Class to find the integrated metal density around a halo.
    Inherits from HaloMet, and finds grids of metal density in amu / cm^3."""
    def __init__(self,snap_dir,snapnum,elem,ion,minpart=400,reload_file=False,savefile=None):
        self.elem=elem
        self.ion=ion
        self.species = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']
        self.minpart=minpart
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        self.set_units()

        if savefile==None:
            self.savefile=path.join(snap_dir,"snapdir_"+str(snapnum).rjust(3,'0'),"halomet_grid.hdf5")
        else:
            self.savefile = savefile
        try:
            if reload_file:
                raise KeyError("reloading")
            #First try to load from a file
            self.load_savefile(self.savefile)
        except (IOError,KeyError):
            self.load_header()
            self.load_halos(minpart)
            #Generate cloudy tables
            self.cloudy_table = convert_cloudy.CloudyTable(self.redshift)
            # Conversion factors from internal units
            rscale = self.UnitLength_in_cm/(1+self.redshift)/self.hubble    # convert length to cm
            mscale = self.UnitMass_in_g/self.hubble   # convert mass to g
            self.dscale = mscale / rscale **3 # Convert density to g / cm^3
            #Otherwise regenerate from the raw data
            self.sub_nHI_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
            self.set_nHI_grid()
        return

    def set_nHI_grid(self, gas=False):
        """Set up the grid around each halo where the HI is calculated.
        """
        self.once=True
        #Now grid the HI for each halo
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
        files.reverse()
        for ff in files:
            f = h5py.File(ff,"r")
            print "Starting file ",ff
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"])
            smooth = hsml.get_smooth_length(bar)
            [self.sub_gridize_single_file(ii,ipos,smooth,bar,self.sub_nHI_grid) for ii in xrange(0,self.nhalo)]
            f.close()
            #Explicitly delete some things.
            del ipos
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

    def sub_gridize_single_file(self,ii,ipos,ismooth,bar,sub_nHI_grid,weights=None):
        """Helper function for sub_nHI_grid
            that puts data arrays loaded from a particular file onto the grid.
            Arguments:
                pos - Position array
                rho - Density array to be interpolated
                smooth - Smoothing lengths
                sub_grid - Grid to add the interpolated data to
        """
        #Find particles near each halo
        sub_pos=self.sub_cofm[ii]
        grid_radius = self.sub_radii[ii]
        #Need a local for numexpr
        box = self.box
        #Get gas mass in internal units
        mass=np.array(bar["Masses"])
        #Density in this species
        nelem = self.species.index(self.elem)
        mass_frac=np.array(bar["GFM_Metals"][:,nelem])
        #In g/cm^3
        den = np.array(bar["Density"])*self.dscale
        #In (hydrogen) atoms / cm^3
        den /= self.protonmass
        #Gather all nearby cells, paying attention to periodic box conditions
        for dim in np.arange(3):
            jpos = sub_pos[dim]
            jjpos = ipos[:,dim]
            indj = np.where(ne.evaluate("(abs(jjpos-jpos) < grid_radius+ismooth) | (abs(jjpos-jpos+box) < grid_radius+ismooth) | (abs(jjpos-jpos-box) < grid_radius+ismooth)"))

            if np.size(indj) == 0:
                return

            ipos = ipos[indj]

            # Update smooth and rho arrays as well:
            ismooth = ismooth[indj]
            mass = mass[indj]
            mass_frac = mass_frac[indj]
            den = den[indj]

            jjpos = ipos[:,dim]
            # BC 1:
            ind_bc1 = np.where(ne.evaluate("(abs(jjpos-jpos+box) < grid_radius+ismooth)"))
            ipos[ind_bc1,dim] = ipos[ind_bc1,dim] + box
            # BC 2:
            ind_bc2 = np.where(ne.evaluate("(abs(jjpos-jpos-box) < grid_radius+ismooth)"))
            ipos[ind_bc2,dim] = ipos[ind_bc2,dim] - box

            #if np.size(ind_bc1)>0 or np.size(ind_bc2)>0:
            #    print "Fixed some periodic cells!"

        if np.size(ipos) == 0:
            return
        mass_frac *= self.cloudy_table.ion(self.elem, self.ion, mass, den)

        #coords in grid units
        coords=fieldize.convert_centered(ipos-sub_pos,self.ngrid[ii],2*self.sub_radii[ii])
        #NH0
        cellspkpc=(self.ngrid[ii]/(2*self.sub_radii[ii]))
        #Convert smoothing lengths to grid coordinates.
        ismooth*=cellspkpc
        if self.once:
            avgsmth=np.mean(ismooth)
            print ii," Av. smoothing length is ",avgsmth/cellspkpc," kpc/h ",avgsmth, "grid cells min: ",np.min(ismooth)
            self.once=False
        #interpolate the density
        fieldize.sph_str(coords,mass*mass_frac,sub_nHI_grid[ii],ismooth,weights=weights)
        return

class BoxMet(bi.BoxHI):
    """Class to find the integrated metal density in a box grid.
    Inherits from BoxHI"""
    def set_nHI_grid(self, gas=False):
        """Set up the grid around each halo where the HI is calculated.
        """
        self.once=True
        #Now grid the HI for each halo
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
        files.reverse()
        for ff in files:
            f = h5py.File(ff,"r")
            print "Starting file ",ff
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"])
            #Get HI mass in internal units
            mass=np.array(bar["Masses"])
            mass *= np.array(bar["GFM_Metallicity"])
            smooth = hsml.get_smooth_length(bar)
            [self.sub_gridize_single_file(ii,ipos,smooth,mass,self.sub_nHI_grid) for ii in xrange(0,self.nhalo)]
            f.close()
            #Explicitly delete some things.
            del ipos
            del mass
            del smooth
        #we calculated things in internal gadget /cell and we want atoms/cm^2
        #So the conversion is mass/(cm/cell)^2
        for ii in xrange(0,self.nhalo):
            massg=self.UnitMass_in_g/self.hubble*self.hy_mass/self.protonmass
            epsilon=2.*self.sub_radii[ii]/(self.ngrid[ii])*self.UnitLength_in_cm/self.hubble/(1+self.redshift)
            self.sub_nHI_grid[ii]*=(massg/epsilon**2)
            self.sub_nHI_grid[ii]+=0.1
            np.log10(self.sub_nHI_grid[ii],self.sub_nHI_grid[ii])
        return

