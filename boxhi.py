# -*- coding: utf-8 -*-
"""Derived class for computing the integrated HI across the whole box.
"""
import numpy as np
import os.path as path
import fieldize
import numexpr as ne
import cold_gas
import halocat
from halohi import HaloHI,calc_binned_median,calc_binned_percentile
import hdfsim
from brokenpowerfit import powerfit
import h5py
import hsml
from _fieldize_priv import _find_halo_kernel,_Discard_SPH_Fieldize

class BoxHI(HaloHI):
    """Class for calculating a large grid encompassing the whole simulation.
    Stores a big grid projecting the neutral hydrogen along the line of sight for the whole box.

    Parameters:
        dir - Simulation directory
        snapnum - Number of simulation
        reload_file - Ignore saved files if true
        nslice - number of slices in the z direction to divide the box into.
    """
    def __init__(self,snap_dir,snapnum,nslice=1,reload_file=False,savefile=None,gas=False, molec=True, start=0, end=3000, ngrid=16384):
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        self.molec = molec
        self.set_units()
        self.start = int(start)
        self.end = int(end)
        if savefile==None:
            savefile = "boxhi_grid_H2.hdf5"
        self.savefile = path.join(snap_dir,"snapdir_"+str(snapnum).rjust(3,'0'),savefile)
        self.tmpfile = self.savefile+"."+str(self.start)+".tmp"
        if gas:
            self.tmpfile+=".gas"
        self.sub_mass = 10.**12*np.ones(nslice)
        self.nhalo = nslice
        if reload_file:
            #Otherwise regenerate from the raw data
            self.load_header()
            #global grid in slices
            self.sub_cofm=0.5*np.ones([nslice,3])
            self.sub_cofm[:,0]=(np.arange(0,nslice)+0.5)/(1.*nslice)*self.box
            self.sub_radii=self.box/2.*np.ones(nslice)
            #Grid size double softening length
            #self.ngrid=np.array([int(np.ceil(40*self.npart[1]**(1./3)/self.box*2*rr)) for rr in self.sub_radii])/2.
            #Grid size constant
            self.ngrid=ngrid*np.ones(self.nhalo)
            self.sub_nHI_grid=np.zeros([self.nhalo, ngrid,ngrid])
            try:
                thisstart = self.load_tmp()
            except IOError:
                print "Could not load file"
                thisstart = self.start
            self.set_nHI_grid(gas, thisstart)
            #Account for molecular fraction
            #This is done on the HI density now
            #self.set_stellar_grid()
            #+ because we are in log space
            #self.sub_nHI_grid+=np.log10(1.-self.h2frac(10**self.sub_nHI_grid, self.sub_star_grid))
        else:
            #try to load from a file
            self.load_savefile(self.savefile)
        return

    def save_file(self, save_grid=False, LLS_cut = 17., DLA_cut = 20.3):
        """Save the file, by default without the grid"""
        HaloHI.save_file(self,save_grid)
        #Save a list of DLA positions instead
        f=h5py.File(self.savefile,'r+')
        ind = np.where(self.sub_nHI_grid > DLA_cut)
        ind_LLS = np.where((self.sub_nHI_grid > LLS_cut)*(self.sub_nHI_grid < DLA_cut))
        grp = f.create_group("abslists")
        grp.create_dataset("DLA",data=ind)
        grp.create_dataset("DLA_val",data=self.sub_nHI_grid[ind])
        grp.create_dataset("LLS",data=ind_LLS)
        grp.create_dataset("LLS_val",data=self.sub_nHI_grid[ind_LLS])
        f.close()

    def _find_particles_in_slab(self,ii,ipos,ismooth, mHI):
        """Find particles in the slab and convert their units to grid units"""
        #At the moment any particle which is located in the slice is wholly in the slice:
        #no periodicity or smoothing.
        #Gather all particles in this slice
        #Is the avg smoothing length is ~100 kpc and the slice is ~2.5 Mpc wide, this will be a small effect.
        jpos = self.sub_cofm[ii,0]
        jjpos = ipos[:,0]
        grid_radius = self.box/self.nhalo/2.
        indj = np.where(ne.evaluate("abs(jjpos-jpos) < grid_radius"))

        if np.size(indj) == 0:
            return (None, None, None)

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
        return (coords, ismooth*cellspkpc, mHI)

    def sub_gridize_single_file(self,ii,ipos,ismooth,mHI,sub_nHI_grid,weights=None):
        """Helper function for sub_nHI_grid
            that puts data arrays loaded from a particular file onto the grid.
            Arguments:
                pos - Position array
                rho - Density array to be interpolated
                smooth - Smoothing lengths
                sub_grid - Grid to add the interpolated data to
        """
        (coords, ismooth, mHI) = self._find_particles_in_slab(ii,ipos,ismooth, mHI)
        if coords == None:
            return
        fieldize.sph_str(coords,mHI,sub_nHI_grid[ii],ismooth,weights=weights, periodic=True)
        return

    def set_stellar_grid(self):
        """Set up a grid around each halo containing the stellar column density
        """
        self.sub_star_grid=np.zeros([self.nhalo, self.ngrid[0],self.ngrid[0]])
        self.once=True
        #Now grid the HI for each halo
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
        files.reverse()
        for ff in files:
            f = h5py.File(ff,"r")
            print "Starting file ",ff
            bar=f["PartType4"]
            ipos=np.array(bar["Coordinates"])
            #Get HI mass in internal units
            mass=np.array(bar["Masses"])
            smooth = np.array(bar["SubfindHsml"])
            [self.sub_gridize_single_file(ii,ipos,smooth,mass,self.sub_star_grid) for ii in xrange(0,self.nhalo)]
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
            self.sub_star_grid[ii]*=(massg/epsilon**2)
        return

    def load_fast_tmp(self,start,key):
        return start
    def save_fast_tmp(self,location,key):
        return

    def set_zdir_grid(self, dlaind, gas=False, key="zpos", ion=-1):
        """Set up the grid around each halo where the HI is calculated.
        """
        star=cold_gas.RahmatiRT(self.redshift, self.hubble, molec=self.molec)
        self.once=True
        #Now grid the HI for each halo
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
        files.reverse()
        self.xslab = np.zeros_like(dlaind[0], dtype=np.float64)
        try:
            start = self.load_fast_tmp(self.start, key)
        except IOError:
            start = self.start
        end = np.min([np.size(files),self.end])
        for xx in xrange(start,end):
            ff = files[xx]
            f = h5py.File(ff,"r")
            print "Starting file ",ff
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"])
            #Get HI mass in internal units
            mass=np.array(bar["Masses"])
            if not gas:
                #Hydrogen mass fraction
                try:
                    mass *= np.array(bar["GFM_Metals"][:,0])
                except KeyError:
                    mass *= self.hy_mass
            nhi = star.get_reproc_HI(bar)
            ind = np.where(nhi > 1.e-3)
            ipos = ipos[ind,:][0]
            mass = mass[ind]
            #Get x * m for the weighted z direction
            if not gas:
                mass *= nhi[ind]
            if key == "zpos":
                mass*=ipos[:,0]
            elif key != "":
                mass *= self._get_secondary_array(ind,bar,key, ion)
            smooth = hsml.get_smooth_length(bar)[ind]
            for slab in xrange(self.nhalo):
                ind = np.where(dlaind[0] == slab)
                self.xslab[ind] += self.sub_list_grid_file(slab,ipos,smooth,mass,dlaind[1][ind], dlaind[2][ind])

            f.close()
            #Explicitly delete some things.
            del ipos
            del mass
            del smooth
            self.save_fast_tmp(start,key)

        #Fix the units:
        #we calculated things in internal gadget /cell and we want atoms/cm^2
        #So the conversion is mass/(cm/cell)^2
        massg=self.UnitMass_in_g/self.hubble/self.protonmass
        epsilon=2.*self.sub_radii[0]/(self.ngrid[0])*self.UnitLength_in_cm/self.hubble/(1+self.redshift)
        self.xslab*=(massg/epsilon**2)
        return self.xslab

    def _get_secondary_array(self, ind, bar, key, ion=1):
        """Get the array whose HI weighted amount we want to compute. Throws ValueError
        if key is not a desired species."""
        raise NotImplementedError("Not valid species")

    def sub_list_grid_file(self,ii,ipos,ismooth,mHI,yslab, zslab):
        """Like sub_gridize_single_file for set_zdir_grid
        """
        (coords, ismooth, mHI) = self._find_particles_in_slab(ii,ipos,ismooth, mHI)
        if coords == None:
            return np.zeros_like(yslab)

        slablist = yslab*int(self.ngrid[0])+zslab
        xslab = _Discard_SPH_Fieldize(slablist, coords, ismooth, mHI, np.array([0.]),True,int(self.ngrid[0]))
        return xslab

    def absorption_distance(self):
        """Compute X(z), the absorption distance per sightline (eq. 9 of Nagamine et al 2003)
        in dimensionless units, accounting for slicing the box."""
        #h * 100 km/s/Mpc in h/s
        h100=3.2407789e-18
        # in cm/s
        light=2.9979e10
        #Units: h/s   s/cm                        kpc/h      cm/kpc
        return h100/light*(1+self.redshift)**2*(self.box/self.nhalo)*self.UnitLength_in_cm


    def omega_DLA(self, thresh=20.3, upthresh=50., fact=1000):
        """Compute Omega_DLA, the sum of the neutral gas in DLAs, divided by the critical density.
            Ω_DLA =  m_p * avg. column density / (1+z)^2 / length of column / rho_c / X_H
            Note: If we want the neutral hydrogen density rather than the gas hydrogen density, multiply by 0.76,
            the hydrogen mass fraction.
            The Noterdaeme results are GAS MASS
        """
        try:
            return self.Omega_DLA
        except AttributeError:
            #Avg density in g/cm^3 (comoving) divided by critical density in g/cm^3
            omega_DLA=fact*self._rho_DLA(thresh, upthresh)/self.rho_crit()
            self.Omega_DLA = omega_DLA
            return omega_DLA

    def _rho_DLA(self, thresh=20.3, upthresh=50.):
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
        HImass = self.protonmass * HImass/(1+self.redshift)**2
        #Length of column in comoving cm
        length = (self.box*self.UnitLength_in_cm/self.hubble/self.nhalo)
        #Avg density in g/cm^3 (comoving)
        return HImass/length

    def get_omega_hi_mass_breakdown(self, rhohi=True):
        """Get the total HI mass density in DLAs in each halo mass bin.
        Returns Omega_DLA in each mass bin."""
        (halo_mass, _, _) = self._load_halo(0)
        self._get_sigma_DLA(0,2)
        ind = np.where(self.dla_halo >= 0)
        find = np.where(self.dla_halo < 0)
        masses = halo_mass[self.dla_halo[ind]]
        dlaval = self._load_dla_val(True)
        massbins = 10**np.arange(9,13)
        massbins[0] = 10**8
        massbins[-1] = 10**13
        nmassbins = np.size(massbins)-1
        fractions = np.zeros(nmassbins+2)
        for mm in xrange(nmassbins):
            mind = np.where((masses > massbins[mm])*(masses <= massbins[mm+1]))
            if rhohi:
                fractions[mm+1] = np.sum(10**dlaval[ind][mind])
            else:
                fractions[mm+1] = np.size(mind)
        #Field DLAs
        if rhohi:
            fractions[nmassbins+1] = np.sum(10**dlaval[find])
        else:
            fractions[nmassbins+1] = np.size(find)
        #Divide by total number of sightlines
        fractions /= (self.nhalo*self.ngrid[0]**2)
        if rhohi:
            #Avg. Column density of HI in g cm^-2 (comoving)
            fractions = self.protonmass * fractions/(1+self.redshift)**2
            #Length of column in comoving cm
            length = (self.box*self.UnitLength_in_cm/self.hubble/self.nhalo)
            #Avg Density in g/cm^3 (comoving) divided by rho_crit
            fractions = 1000*fractions/length/self.rho_crit()
        else:
            fractions /= self.absorption_distance()
        return (massbins, fractions)

    def rho_DLA(self, thresh=20.3):
        """Compute rho_DLA, the sum of the mass in DLAs. This is almost the same as the total mass in HI.
           Units are 10^8 M_sun / Mpc^3 (comoving), like 0811.2003
        """
        try:
            return self.Rho_DLA
        except AttributeError:
            #Avg density in g/cm^3 (comoving) / a^3 = physical
            rho_DLA = self._rho_DLA(thresh)  #*(1.+self.redshift)**3
            # 1 g/cm^3 (physical) in 1e8 M_sun/Mpc^3
            conv = 1e8 * self.SolarMass_in_g / (1e3 * self.UnitLength_in_cm)**3
            self.Rho_DLA = rho_DLA / conv
            return rho_DLA / conv

    def line_density(self, thresh=20.3):
        """Compute the line density, the total cells in DLAs divided by the total area, multiplied by d L / dX. This is dN/dX = l_DLA(z)
        """
        #P(hitting a DLA at random)
        try:
            return self.pDLA
        except AttributeError:
            DLAs = 1.*np.sum(self.sub_nHI_grid > thresh)
            size = 1.*np.sum(self.ngrid**2)
            pDLA = DLAs/size/self.absorption_distance()
            self.pDLA = pDLA
            return pDLA

    def line_density2(self,thresh=20.3):
        """Compute the line density the other way, by summing the cddf. This is dN/dX = l_DLA(z)"""
        (_,  fN) = self.column_density_function(minN=thresh,maxN=24)
        NHI_table = 10**np.arange(thresh, 24, 0.2)
        width =  np.array([NHI_table[i+1]-NHI_table[i] for i in range(0,np.size(NHI_table)-1)])
        return np.sum(fN*width)

    def omega_DLA2(self,thresh=20.3):
        """Compute Omega_DLA the other way, by summing the cddf."""
        (center,  fN) = self.column_density_function(minN=thresh,maxN=24)
        #f_N* NHI is returned, in amu/cm^2/dX
        NHI_table = 10**np.arange(thresh, 24, 0.2)
        width =  np.array([NHI_table[i+1]-NHI_table[i] for i in range(0,np.size(NHI_table)-1)])
        dXdcm = self.absorption_distance()/((self.box/self.nhalo)*self.UnitLength_in_cm/self.hubble)
        return 1000*self.protonmass*np.sum(fN*center*width)*dXdcm/self.rho_crit()/(1+self.redshift)**2


    def save_sigLLS(self):
        """Generate and save sigma_LLS to the savefile"""
        (self.real_sub_mass, self.sigLLS, self.field_lls, self.lls_halo) = self.find_cross_section(False, 0, 2.)
        f=h5py.File(self.savefile,'r+')
        mgrp = f["CrossSection"]
        try:
            del mgrp["sigLLS"]
            del mgrp["LLS_halo"]
        except KeyError:
            pass
        mgrp.attrs["field_lls"] = self.field_lls
        mgrp.create_dataset("sigLLS",data=self.sigLLS)
        mgrp.create_dataset("LLS_halo",data=self.lls_halo)
        f.close()

    def load_sigLLS(self):
        """Load sigma_LLS from a file"""
        f=h5py.File(self.savefile,'r')
        try:
            mgrp = f["CrossSection"]
            self.real_sub_mass = np.array(mgrp["sub_mass"])
            self.sigLLS = np.array(mgrp["sigLLS"])
            self.lls_halo = np.array(mgrp["LLS_halo"])
            self.field_lls = mgrp.attrs["field_lls"]
        except KeyError:
            f.close()
            raise
        f.close()

    def save_sigDLA(self):
        """Generate and save sigma_DLA to the savefile"""
        (self.real_sub_mass, self.sigDLA, self.field_dla,self.dla_halo) = self.find_cross_section(True, 0, 2.)
        f=h5py.File(self.savefile,'r+')
        try:
            mgrp = f.create_group("CrossSection")
        except ValueError:
            mgrp = f["CrossSection"]
        try:
            del mgrp["sub_mass"]
            del mgrp["sigDLA"]
            del mgrp["DLAzdir"]
            del mgrp["DLA_halo"]
        except KeyError:
            pass
        mgrp.attrs["field_dla"] = self.field_dla
        mgrp.create_dataset("sub_mass",data=self.real_sub_mass)
        mgrp.create_dataset("sigDLA",data=self.sigDLA)
        mgrp.create_dataset("DLA_halo",data=self.dla_halo)
        mgrp.create_dataset("DLAzdir",data=self.dla_zdir)
        f.close()

    def load_sigDLA(self):
        """Load sigma_DLA from a file"""
        f=h5py.File(self.savefile,'r')
        try:
            mgrp = f["CrossSection"]
            self.real_sub_mass = np.array(mgrp["sub_mass"])
            self.sigDLA = np.array(mgrp["sigDLA"])
            self.dla_halo = np.array(mgrp["DLA_halo"])
            self.field_dla = mgrp.attrs["field_dla"]
        except KeyError:
            f.close()
            raise
        f.close()

    def find_cross_section(self, dla=True, minpart=0, vir_mult=2):
        """Find the number of DLA cells within dist virial radii of
           each halo resolved with at least minpart particles.
           If within the virial radius of multiple halos, use the most massive one."""
        (halo_mass, halo_cofm, halo_radii) = self._load_halo(minpart)
        dlaind = self._load_dla_index(dla)
        #Computing z distances
        xslab = self._get_dla_zpos(dlaind,dla)
        self.dla_zdir=xslab
        dla_cross = np.zeros_like(halo_mass)
        celsz = 1.*self.box/self.ngrid[0]
        yslab = (dlaind[1]+0.5)*celsz
        zslab = (dlaind[2]+0.5)*celsz
        assigned_halo = np.zeros_like(yslab, dtype=np.int32)
        assigned_halo-=1
        print "Starting find_halo_kernel"
        field_dla = _find_halo_kernel(self.box, halo_cofm,vir_mult*halo_radii,halo_mass, xslab, yslab, zslab,dla_cross, assigned_halo)
        print "max = ",np.max(dla_cross)," field dlas: ",100.*field_dla/np.shape(dlaind)[1]
        #Convert from grid cells to kpc/h^2
        dla_cross*=celsz**2
        return (halo_mass, dla_cross, 100.*field_dla/np.shape(dlaind)[1],assigned_halo)

    def _get_dla_zpos(self,dlaind,dla=True):
        """Load or compute the depth of the DLAs"""
        if dla == False:
            raise NotImplementedError("Does not work for LLS")
        f=h5py.File(self.savefile,'r')
        try:
            xslab = np.array(f["CrossSection"]["DLAzdir"])
        except KeyError:
            xhimass = self.set_zdir_grid(dlaind)
            xslab = xhimass/10**self._load_dla_val(dla)
        f.close()
        return xslab

    def _load_dla_index(self, dla=True):
        """Load the positions of DLAs or LLS from savefile"""
        #Load the DLA/LLS positions
        f=h5py.File(self.savefile,'r')
        grp = f["abslists"]
        #This is needed to make the dimensions right
        if dla:
            ind = (grp["DLA"][0,:],grp["DLA"][1,:],grp["DLA"][2,:])
        else:
            ind = (grp["LLS"][0,:],grp["LLS"][1,:],grp["LLS"][2,:])
        f.close()
        return ind

    def _load_dla_val(self, dla=True):
        """Load the values of DLAs or LLS from savefile"""
        #Load the DLA/LLS positions
        f=h5py.File(self.savefile,'r')
        grp = f["abslists"]
        #This is needed to make the dimensions right
        if dla:
            nhi = np.array(grp["DLA_val"])
        else:
            nhi = np.array(grp["LLS_val"])
        f.close()
        return nhi

    def _load_halo(self, minpart=400):
        """Load a halo catalogue"""
        #This is rho_c in units of h^-1 M_sun (kpc/h)^-3
        rhom = 2.78e+11* self.omegam / (1e3**3)
        #Mass of an SPH particle, in units of M_sun, x omega_m/ omega_b.
        target_mass = self.box**3 * rhom / self.npart[0]
        min_mass = target_mass * minpart / 1e10
        (_, halo_mass, halo_cofm, halo_radii) = halocat.find_all_halos(self.snapnum, self.snap_dir, min_mass)
        return (halo_mass, halo_cofm, halo_radii)

    def column_density_function(self,dlogN=0.1, minN=16, maxN=24., maxM=None,minM=None):
        """
        This computes the DLA column density function, which is the number
        of absorbers per sight line with HI column densities in the interval
        [NHI, NHI+dNHI] at the absorption distance X.
        Absorption distance is simply a single simulation box.
        A sightline is assumed to be equivalent to one grid cell.
        That is, there is presumed to be only one halo in along the sightline
        encountering a given halo.

        So we have f(N) = d n_DLA/ dN dX
        and n_DLA(N) = number of absorbers per sightline in this column density bin.
                     1 sightline is defined to be one grid cell.
                     So this is (cells in this bins) / (no. of cells)
        ie, f(N) = n_DLA / ΔN / ΔX
        Note f(N) has dimensions of cm^2, because N has units of cm^-2 and X is dimensionless.

        Parameters:
            dlogN - bin spacing
            minN - minimum log N
            maxN - maximum log N
            maxM - maximum log M halo mass to consider
            minM - minimum log M halo mass to consider

        Returns:
            (NHI, f_N_table) - N_HI (binned in log) and corresponding f(N)
        """
        NHI_table = 10**np.arange(minN, maxN, dlogN)
        if maxM == None and minM == None:
            try:
                if np.size(NHI_table)-1 == np.size(self.cddf_bins):
                    return (self.cddf_bins, self.cddf_f_N)
                else:
                    raise AttributeError
            except AttributeError:
                (self.cddf_bins, self.cddf_f_N)= self._calc_cddf(NHI_table, minN, maxM, minM)
                return (self.cddf_bins, self.cddf_f_N)
        else:
            return self._calc_cddf(NHI_table, minN, maxM, minM)


    def _calc_cddf(self,NHI_table, minN=17, maxM=None,minM=None):
        """Does the actual calculation for the CDDF function above"""
        try:
            grids = self.sub_nHI_grid
        except AttributeError:
            self.load_hi_grid()
            grids = self.sub_nHI_grid
        center = np.array([(NHI_table[i]+NHI_table[i+1])/2. for i in range(0,np.size(NHI_table)-1)])
        width =  np.array([NHI_table[i+1]-NHI_table[i] for i in range(0,np.size(NHI_table)-1)])
        #Grid size (in cm^2)
        dX=self.absorption_distance()
        tot_cells = np.sum(self.ngrid**2)
        if maxM != None:
            raise NotImplementedError("Splitting by mass no longer works")
            ind = np.where((self.halo_mass < 10.**maxM)*(self.halo_mass > 10.**minM))
            tot_f_N = np.histogram(np.ravel(grids[ind]),np.log10(NHI_table))[0]
        else:
            ind = np.where(grids >= minN)
            tot_f_N = np.histogram(np.ravel(grids[ind]),np.log10(NHI_table))[0]
        tot_f_N=(tot_f_N)/(width*dX*tot_cells)
        return (center, tot_f_N)

    def get_dla_metallicity(self):
        """Get the DLA metallicities from the save file, as Z/Z_sun.
        """
        try:
            return self.dla_metallicity-np.log10(self.solarz)
        except AttributeError:
            ff = h5py.File(self.savefile,"r")
            self.dla_metallicity = np.array(ff["Metallicities"]["DLA"])
            ff.close()
            return self.dla_metallicity-np.log10(self.solarz)

    def get_ion_metallicity(self, species,ion, dla=True):
        """Get the metallicity derived from an ionic species"""
        f=h5py.File(self.savefile,'r')
        grp = f[species][str(ion)]
        #This is needed to make the dimensions right
        if dla:
            spec = np.array(grp["DLA"])
        else:
            spec = np.array(grp["LLS"])
        f.close()
        #Divide by H column density
        hi = self._load_dla_val(dla)
        met = np.log10(spec+0.01)-hi-np.log10(self.solar[species])
        return met

    def get_lls_metallicity(self):
        """Get the LLS metallicities from the save file, as Z/Z_solar
        """
        try:
            return self.lls_metallicity-np.log10(self.solarz)
        except AttributeError:
            ff = h5py.File(self.savefile,"r")
            self.lls_metallicity = np.array(ff["Metallicities"]["LLS"])
            ff.close()
            return self.lls_metallicity-np.log10(self.solarz)

    def get_sDLA_fit(self):
        """Fit an broken power law profile based function to sigma_DLA as binned."""
        minM = np.min(self.real_sub_mass)
        maxM = np.max(self.real_sub_mass)
        bins=30
        mass=np.logspace(np.log10(minM),np.log10(maxM),num=bins)
        bin_mass = np.array([(mass[i]+mass[i+1])/2. for i in xrange(0,np.size(mass)-1)])
        (sDLA,loq,upq)=self.get_sigma_DLA_binned(mass,sigma=68)
        ind = np.where((sDLA > 0)*(loq+upq > 0)*(bin_mass > 10**8.5))
        err = (upq[ind]+loq[ind])/2.
        #Arbitrary large values if err is zero
        return powerfit(np.log10(bin_mass[ind]), np.log10(sDLA[ind]), np.log10(err), breakpoint=10)

    def get_sigma_DLA_binned(self,mass,DLA_cut=20.3,DLA_upper_cut=42.,sigma=95):
        """Get the median and scatter of sigma_DLA against mass."""
        if DLA_cut < 17.5:
            sigs = self.sigLLS
        else:
            sigs = self.sigDLA
        aind = np.where(sigs > 0)
        #plt.loglog(self.real_sub_mass[aind], self.sigDLA[aind],'x')
        amed=calc_binned_median(mass, self.real_sub_mass[aind], sigs[aind])
        aupq=calc_binned_percentile(mass, self.real_sub_mass[aind], sigs[aind],sigma)-amed
        #Addition to avoid zeros
        aloq=amed - calc_binned_percentile(mass, self.real_sub_mass[aind], sigs[aind],100-sigma)
        return (amed, aloq, aupq)

    def _get_sigma_DLA(self, minpart, dist):
        """Helper for halo_hist to correctly populate sigDLA, from a savefile if possible"""
        if minpart == 0 and dist == 2.:
            try:
                self.sigDLA
            except AttributeError:
                try:
                    self.load_sigDLA()
                except KeyError:
                    self.save_sigDLA()
        else:
            (self.real_sub_mass, self.sigDLA, self.field_dla, self.dla_halo) = self.find_cross_section(True, minpart, dist)

    def _get_sigma_LLS(self, minpart, dist):
        """Helper for halo_hist to correctly populate sigLLS, from a savefile if possible"""
        if minpart == 0 and dist == 2.:
            try:
                self.sigLLS
            except AttributeError:
                try:
                    self.load_sigLLS()
                except KeyError:
                    self.save_sigLLS()
        else:
            (self.real_sub_mass, self.sigLLS, self.field_lls, self.lls_halo) = self.find_cross_section(False, minpart, dist)

    def get_dla_impact_parameter(self, minM, maxM):
        """Get the distance from the parent halo as a fraction of rvir for each DLA"""
        (halo_mass, halo_cofm, halo_radii) = self._load_halo(0)
        self._get_sigma_DLA(0,2)
        dlaind = self._load_dla_index(True)
        #Halo positions
        ind = np.where(self.dla_halo >= 0)
        halopos = halo_cofm[self.dla_halo[ind]]
        #Computing z distances
        xslab = self._get_dla_zpos(dlaind,True)
        yslab = (dlaind[1]+0.5)*self.box*1./self.ngrid[0]
        zslab = (dlaind[2]+0.5)*self.box*1./self.ngrid[0]
        #Total distance
        xdist = np.abs(xslab[ind]-halopos[:,0])
        ydist = np.abs(yslab[ind]-halopos[:,1])
        zdist = np.abs(zslab[ind]-halopos[:,2])
        #Deal with periodics
        ii = np.where(xdist > self.box/2.)
        xdist[ii] = self.box-xdist[ii]
        ii = np.where(ydist > self.box/2.)
        ydist[ii] = self.box-ydist[ii]
        ii = np.where(zdist > self.box/2.)
        zdist[ii] = self.box-zdist[ii]
        distance = np.sqrt(xdist**2 + ydist**2 + zdist**2)

        ind2 = np.where((halo_mass[self.dla_halo[ind]] > minM)*(halo_mass[self.dla_halo[ind]] < maxM))
        return (distance/halo_radii[self.dla_halo[ind]])[ind2]
