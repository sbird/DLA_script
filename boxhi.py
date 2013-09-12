# -*- coding: utf-8 -*-
"""Derived class for computing the integrated HI across the whole box.
"""
import numpy as np
import os.path as path
import fieldize
import numexpr as ne
import halocat
from halohi import HaloHI,calc_binned_median,calc_binned_percentile
import hdfsim
from brokenpowerfit import powerfit
import h5py
from _fieldize_priv import _find_halo_kernel

class BoxHI(HaloHI):
    """Class for calculating a large grid encompassing the whole simulation.
    Stores a big grid projecting the neutral hydrogen along the line of sight for the whole box.

    Parameters:
        dir - Simulation directory
        snapnum - Number of simulation
        reload_file - Ignore saved files if true
        nslice - number of slices in the z direction to divide the box into.
    """
    def __init__(self,snap_dir,snapnum,nslice=1,reload_file=False,savefile=None,gas=False, molec=True):
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        self.molec = molec
        self.set_units()
        if savefile==None:
            self.savefile = path.join(snap_dir,"snapdir_"+str(snapnum).rjust(3,'0'),"boxhi_grid_H2.hdf5")
        else:
            self.savefile = savefile
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
            self.ngrid=16384*np.ones(self.nhalo)
            self.sub_nHI_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
            self.set_nHI_grid(gas)
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

    def set_stellar_grid(self):
        """Set up a grid around each halo containing the stellar column density
        """
        self.sub_star_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
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
            omega_DLA=1000*self._rho_DLA(thresh)/self.rho_crit()
            self.Omega_DLA = omega_DLA
            return omega_DLA

    def _rho_DLA(self, thresh=20.3):
        """Find the average density in DLAs in g/cm^3 (comoving). Helper for omega_DLA and rho_DLA."""
        #Average column density of HI in atoms cm^-2 (physical)
        try:
            self.sub_nHI_grid
        except AttributeError:
            self.load_hi_grid()
        if thresh > 0:
            grids=self.sub_nHI_grid
            HImass = np.sum(10**grids[np.where(grids > thresh)])/np.size(grids)
        else:
            HImass = np.mean(10**self.sub_nHI_grid)
        #Avg. Column density of HI in g cm^-2 (comoving)
        HImass = self.protonmass * HImass/(1+self.redshift)**2
        #Length of column in comoving cm
        length = (self.box*self.UnitLength_in_cm/self.hubble/self.nhalo)
        #Avg density in g/cm^3 (comoving)
        return HImass/length

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

    def save_sigDLA(self):
        """Generate and save sigma_DLA to the savefile"""
        (self.real_sub_mass, self.sigDLA, self.field_dla) = self.find_cross_section(True, 0, 1.)
        f=h5py.File(self.savefile,'r+')
        try:
            mgrp = f.create_group("CrossSection")
        except ValueError:
            mgrp = f["CrossSection"]
        try:
            del mgrp["sub_mass"]
            del mgrp["sigDLA"]
        except KeyError:
            pass
        mgrp.attrs["field_dla"] = self.field_dla
        mgrp.create_dataset("sub_mass",data=self.real_sub_mass)
        mgrp.create_dataset("sigDLA",data=self.sigDLA)
        f.close()

    def load_sigDLA(self,savefile=None):
        """Load sigma_DLA from a file"""
        f=h5py.File(self.savefile,'r')
        try:
            mgrp = f["CrossSection"]
        except KeyError:
            f.close()
            raise
        self.real_sub_mass = np.array(mgrp["sub_mass"])
        self.sigDLA = np.array(mgrp["sigDLA"])
        self.field_dla = mgrp.attrs["field_dla"]
        f.close()

    def find_cross_section(self, dla=True, minpart=0, vir_mult=1.):
        """Find the number of DLA cells within dist virial radii of
           each halo resolved with at least minpart particles.
           If within the virial radius of multiple halos, use the most massive one."""
        (halo_mass, halo_cofm, halo_radii) = self._load_halo(minpart)
        dlaind = self._load_dla_index(dla)
        dla_cross = np.zeros_like(halo_mass)
        celsz = 1.*self.box/self.ngrid[0]
        field_dla = 0
        #Treat each slice separately
        for ss in xrange(self.nhalo):
            halo_ind = np.where(np.logical_and(halo_cofm[:,0] > ss*self.box/self.nhalo, halo_cofm[:,0] < (ss+1)*self.box/self.nhalo))
            slab_ind = np.where(dlaind[0] == ss)
            yslab = (dlaind[1][slab_ind]+0.5)*self.box*1./self.ngrid[0]
            zslab = (dlaind[2][slab_ind]+0.5)*self.box*1./self.ngrid[0]
            field_dla += _find_halo_kernel(self.box, halo_cofm[halo_ind,1:][0],halo_mass[halo_ind],vir_mult*halo_radii[halo_ind],yslab, zslab,dla_cross)
            print "Done slab ",ss,"field_dla = ",field_dla

        print "max = ",np.max(dla_cross)," field dlas: ",100.*field_dla/np.shape(dlaind)[1]
        #Convert from grid cells to kpc/h^2
        dla_cross*=celsz**2
        return (halo_mass, dla_cross, 100.*field_dla/np.shape(dlaind)[1])

    def _load_dla_index(self, dla=True):
        """Load the positions of DLAs or LLS from savefile"""
        #Load the DLA/LLS positions
        f=h5py.File(self.savefile,'r+')
        grp = f["abslists"]
        #This is needed to make the dimensions right
        if dla:
            ind = (grp["DLA"][0,:],grp["DLA"][1,:],grp["DLA"][2,:])
        else:
            ind = (grp["LLS"][0,:],grp["LLS"][1,:],grp["LLS"][2,:])
        f.close()
        return ind

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

    def get_dla_metallicity(self, solar=0.0133):
        """Get the DLA metallicities from the save file, as [M/H]"""
        try:
            return self.dla_metallicity-np.log10(solar)
        except AttributeError:
            ff = h5py.File(self.savefile,"r")
            self.dla_metallicity = np.array(ff["Metallicities"]["DLA"])
            return self.dla_metallicity-np.log10(solar)

    def get_lls_metallicity(self, solar=0.0133):
        """Get the LLS metallicities from the save file, as [M/H]"""
        try:
            return self.lls_metallicity-np.log10(solar)
        except AttributeError:
            ff = h5py.File(self.savefile,"r")
            self.lls_metallicity = np.array(ff["Metallicities"]["LLS"])
            return self.lls_metallicity-np.log10(solar)

    def get_sDLA_fit(self):
        """Fit an broken power law profile based function to sigma_DLA as binned."""
        minM = np.min(self.real_sub_mass)
        maxM = np.max(self.real_sub_mass)
        bins=30
        mass=np.logspace(np.log10(minM),np.log10(maxM),num=bins)
        bin_mass = np.array([(mass[i]+mass[i+1])/2. for i in xrange(0,np.size(mass)-1)])
        (sDLA,loq,upq)=self.get_sigma_DLA_binned(mass,sigma=68)
        ind = np.where((sDLA > 0)*(loq+upq > 0)*(bin_mass > 3e10))
        err = (upq[ind]+loq[ind])/2.
        #Arbitrary large values if err is zero
        return powerfit(np.log10(bin_mass[ind]), np.log10(sDLA[ind]), np.log10(err), breakpoint=10)

    def get_sigma_DLA_binned(self,mass,DLA_cut=20.3,DLA_upper_cut=42.,sigma=95):
        """Get the median and scatter of sigma_DLA against mass."""
        aind = np.where(self.sigDLA > 0)
        #plt.loglog(self.real_sub_mass[aind], self.sigDLA[aind],'x')
        amed=calc_binned_median(mass, self.real_sub_mass[aind], self.sigDLA[aind])
        aupq=calc_binned_percentile(mass, self.real_sub_mass[aind], self.sigDLA[aind],sigma)-amed
        #Addition to avoid zeros
        aloq=amed - calc_binned_percentile(mass, self.real_sub_mass[aind], self.sigDLA[aind],100-sigma)
        return (amed, aloq, aupq)

