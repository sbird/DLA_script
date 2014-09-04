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
import cold_gas
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
            escale = 1.0e6           # convert energy/unit mass to J kg^-1
            #convert U (J/kg) to T (K) : U = N k T / (γ - 1)
            #T = U (γ-1) μ m_P / k_B
            #where k_B is the Boltzmann constant
            #γ is 5/3, the perfect gas constant
            #m_P is the proton mass
            #μ is 1 / (mean no. molecules per unit atomic weight) calculated in loop.
            boltzmann = 1.3806504e-23
            self.tscale = ((5./3.-1.0) * 1e-3*self.protonmass * escale ) / boltzmann

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
        #Mean molecular weight:
        # \mu = 1 / molecules per unit atomic weight
        #     = 1 / (X + Y /4 + E)
        #     where E = Ne * X, and Y = (1-X).
        #     Can neglect metals as they are heavy.
        #     Leading contribution is from electrons, which is already included
        #     [+ Z / (12->16)] from metal species
        #     [+ Z/16*4 ] for OIV from electrons.
        mu = 1.0/(0.76*(0.75+np.array(bar["ElectronAbundance"])) + 0.25)
        temp = np.array(bar["InternalEnergy"])*self.tscale*mu
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
        mass_frac *= self.cloudy_table.ion(self.elem, self.ion, den, temp)

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
    """
    Class to find the mass-weighted metallicity for a box.
    Inherits from BoxHI
    """
    def __init__(self,snap_dir,snapnum,nslice=1,savefile=None, start=0, end=3000, ngrid=16384):
        bi.BoxHI.__init__(self, snap_dir, snapnum, nslice, False, savefile, False,start=start, end=end,ngrid=ngrid)
        self.sub_ZZ_grid=np.zeros([nslice, ngrid,ngrid])
        try:
            thisstart = self.load_met_tmp(self.start)
        except (IOError,KeyError):
            print "Could not load file"
            thisstart = self.start
        self.set_ZZ_grid(thisstart)

        #Find the metallicity
        for ii in xrange(0, self.nhalo):
            self.sub_ZZ_grid[ii] -= self.sub_nHI_grid[ii]

    def set_ZZ_grid(self, start=0):
        """Set up the mass * metallicity grid for the box
        Same as set_nHI_grid except mass is multiplied by GFM_Metallicity.
        """
        self.once=True
        #Now grid the HI for each halo
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
        files.reverse()
        restart = 10
        end = np.min([np.size(files),self.end])
        for xx in xrange(start, end):
            ff = files[xx]
            f = h5py.File(ff,"r")
            print "Starting file ",ff
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"])
            #Get HI mass in internal units
            mass=np.array(bar["Masses"])
            #Sometimes the metallicity is less than zero: fix that
            met = np.array(bar["GFM_Metallicity"])
            met[np.where(met <=0)] = 1e-50
            mass *= met
            smooth = hsml.get_smooth_length(bar)
            [self.sub_gridize_single_file(ii,ipos,smooth,mass,self.sub_ZZ_grid) for ii in xrange(0,self.nhalo)]
            f.close()
            #Explicitly delete some things.
            del ipos
            del mass
            del smooth
            if xx % restart == 0 or xx == end-1:
                self.save_met_tmp(xx)

        #Deal with zeros: 0.1 will not even register for things at 1e17.
        #Also fix the units:
        #we calculated things in internal gadget /cell and we want atoms/cm^2
        #So the conversion is mass/(cm/cell)^2
        for ii in xrange(0,self.nhalo):
            massg=self.UnitMass_in_g/self.hubble/self.protonmass
            epsilon=2.*self.sub_radii[ii]/(self.ngrid[ii])*self.UnitLength_in_cm/self.hubble/(1+self.redshift)
            self.sub_ZZ_grid[ii]*=(massg/epsilon**2)
            self.sub_ZZ_grid[ii]+=0.1
            np.log10(self.sub_ZZ_grid[ii],self.sub_ZZ_grid[ii])
        return

    def save_file(self):
        """This does something a little perverse: open up self.savefile
        and save the metallicity values only for the indices found in the abslists group"""
        f=h5py.File(self.savefile,'r+')
        grp = f["abslists"]
        #This is needed to make the dimensions right
        dlaind = (grp["DLA"][0,:],grp["DLA"][1,:],grp["DLA"][2,:])
        llsind = (grp["LLS"][0,:],grp["LLS"][1,:],grp["LLS"][2,:])
        mgrp = f.create_group("Metallicities")
        mgrp.create_dataset("DLA",data=self.sub_ZZ_grid[dlaind])
        mgrp.create_dataset("LLS",data=self.sub_ZZ_grid[llsind])
        f.close()

    def save_met_tmp(self, location):
        """Save a partially completed file"""
        f = h5py.File(self.savefile+"."+str(self.start)+".met.tmp",'w')
        grp_grid = f.create_group("GridZZData")
        for i in xrange(0,self.nhalo):
            grp_grid.create_dataset(str(i),data=self.sub_ZZ_grid[i])
        f.attrs["met_file"]=location
        f.close()

    def load_met_tmp(self, start):
        """
        Load a partially completed file
        """
        print self.savefile+"."+str(start)+".met.tmp"
        f = h5py.File(self.savefile+"."+str(start)+".met.tmp",'r')
        grp = f["GridZZData"]
        [ grp[str(i)].read_direct(self.sub_ZZ_grid[i]) for i in xrange(0,self.nhalo)]
        location = f.attrs["met_file"]
        f.close()
        print "Successfully loaded metals from tmp file. Next to do is:",location+1
        return location+1

class FastBoxMet(bi.BoxHI):
    """
    Class to find the mass-weighted metallicity for a box.
    Inherits from BoxHI
    """
    def __init__(self,snap_dir,snapnum,nslice=1,savefile=None, start=0, end=3000, ngrid=16384, cdir=None):
        bi.BoxHI.__init__(self, snap_dir, snapnum, nslice, False, savefile, False,start=start, end=end,ngrid=ngrid)
        if cdir != None:
            self.cloudy_table = convert_cloudy.CloudyTable(self.redshift, cdir)
        else:
            self.cloudy_table = convert_cloudy.CloudyTable(self.redshift)

    def set_ZZ_fast_dla(self, dla=True):
        """Faster metallicity computation for only those cells with a DLA"""
        dlaind = self._load_dla_index(dla)
        #Computing z distances
        xhmass = self.set_zdir_grid(dlaind,gas=True, key="met")
        hmass = self.set_zdir_grid(dlaind,gas=True,key="")
        met = xhmass/hmass
        f=h5py.File(self.savefile,'r+')
        mgrp = f.create_group("Metallicities")
        if dla:
            mgrp.create_dataset("DLA",data=met)
        else:
            mgrp.create_dataset("LLS",data=met)
        f.close()

    def set_metal_species_fast_dla(self, elem, ion, dla=True):
        """Faster metallicity computation for only those cells with a DLA"""
        dlaind = self._load_dla_index(dla)
        #Computing z distances
        species = self.set_zdir_grid(dlaind,gas=True, key=elem, ion=ion)
        f=h5py.File(self.savefile,'r+')
        try:
            mgrp = f.create_group(elem)
            mmgrp = mgrp.create_group(str(ion))
        except ValueError:
            mmgrp = f[elem][str(ion)]
        if dla:
            datas="DLA"
        else:
            datas="LLS"
        try:
            del mmgrp[datas]
        except KeyError:
            pass
        mmgrp.create_dataset(datas,data=species)
        f.close()

    def _get_secondary_array(self, ind, bar, elem="", ion=-1):
        """Get the array whose HI weighted amount we want to compute. Throws ValueError
        if key is not a desired species. Note this saves the total projected mass of each species in
        atoms of that species / cm ^2. If you want the total mass in the species, multiply by its atomic mass."""
        if elem == "met":
            met = np.array(bar["GFM_Metallicity"])[ind]
        else:
            nelem = self.species.index(elem)
            met = np.array(bar["GFM_Metals"][:,nelem])[ind]
            #What is saved is the column density in amu, we want the column density,
            #which is in atoms. So there is a factor of mass.
            met /= self.amasses[elem]
            if ion != -1:
                star=cold_gas.RahmatiRT(self.redshift, self.hubble)
                den=star.get_code_rhoH(bar)
                temp = star.get_temp(bar)
                temp = temp[ind]
                den = den[ind]
                met *= self.cloudy_table.ion(elem, ion, den, temp)
        met[np.where(met <=0)] = 1e-50
        return met

class BoxCIV(bi.BoxHI):
    """
    Class to find omega_CIV for a box.
    Inherits from BoxHI
    """
    def __init__(self,snap_dir,snapnum,nslice=1,reload_file=True, savefile=None, start=0, end=3000, ngrid=16384):
        bi.BoxHI.__init__(self, snap_dir, snapnum, nslice, reload_file=reload_file, savefile=savefile, start=start, end=end,ngrid=ngrid)

    def set_nHI_grid(self, gas=False, start=0):
        """Set up the grid around each halo where the HI is calculated.
        """
        star=cold_gas.RahmatiRT(self.redshift, self.hubble, molec=self.molec)
        self.cloudy_table = convert_cloudy.CloudyTable(self.redshift)
        self.once=True
        #Now grid the HI for each halo
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
        files.reverse()
        end = np.min([np.size(files),self.end])
        for xx in xrange(start, end):
            ff = files[xx]
            f = h5py.File(ff,"r")
            print "Starting file ",ff
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"])
            #Get HI mass in internal units
            mass=np.array(bar["Masses"])
            #Carbon mass fraction
            den = star.get_code_rhoH(bar)
            temp = star.get_temp(bar)
            mass_frac = np.array(bar["GFM_Metals"][:,2])
            #Floor on the mass fraction of the metal
            ind = np.where(mass_frac > 1e-10)
            mass = mass[ind]*mass_frac[ind]
            #High densities will have no CIV anyway.
            den[np.where(den > 1e4)] = 9999.
            den[np.where(den < 1e-7)] = 1.01e-7
            temp[np.where(temp > 3e8)] = 3e8
            temp[np.where(temp < 1e3)] = 1e3
            mass *= self.cloudy_table.ion("C", 4, den[ind], temp[ind])
            smooth = hsml.get_smooth_length(bar)[ind]
            ipos = ipos[ind,:][0]
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
            massg=self.UnitMass_in_g/self.hubble/(self.protonmass*12.011)
            epsilon=2.*self.sub_radii[ii]/(self.ngrid[ii])*self.UnitLength_in_cm/self.hubble/(1+self.redshift)
            self.sub_nHI_grid[ii]*=(massg/epsilon**2)
            self.sub_nHI_grid[ii]+=0.1
            np.log10(self.sub_nHI_grid[ii],self.sub_nHI_grid[ii])
        return

    def _rho_DLA(self, thresh=14, upthresh=50.):
        """Find the average density in DLAs in g/cm^3 (comoving). Helper for omega_DLA and rho_DLA."""
        #Average column density of HI in atoms cm^-2 (physical)
        try:
            self.sub_nHI_grid
        except AttributeError:
            self.load_hi_grid()
        if thresh > 0:
            grids=self.sub_nHI_grid
            HImass = np.sum(10**grids[np.where((grids < upthresh)*(grids > thresh))])/np.size(grids)
        else:
            HImass = np.mean(10**self.sub_nHI_grid)
        #Avg. Column density of HI in g cm^-2 (comoving)
        HImass = 12.011*self.protonmass * HImass/(1+self.redshift)**2
        #Length of column in comoving cm
        length = (self.box*self.UnitLength_in_cm/self.hubble/self.nhalo)
        #Avg density in g/cm^3 (comoving)
        return HImass/length

class HaloCIV(hi.HaloHI, BoxCIV):
    """Plots of the CIV around a single halo"""
    def __init__(self,snap_dir,snapnum,reload_file=True, savefile=None, start=0, end=3000):
        hi.HaloHI.__init__(self,snap_dir,snapnum,minpart=400,reload_file=reload_file,savefile=savefile, gas=False, molec=True, start=start, end = end)

    def set_nHI_grid(self, gas=False, start=0):
        return BoxCIV.set_nHI_grid(self,gas,start)
