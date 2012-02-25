"""Module for creating the DLA hydrogen density plots, as found in Tescari & Viel, 
and Nagamine, Springel and Hernquist, 2003. 

Classes: 
    TotalHaloHI - Finds the average HI fraction in a halo
    HaloHI - Creates a grid around the halo center with the HI fraction calculated at each grid cell
"""
import numpy as np
import re
import readsubf
import hdfsim
import cold_gas
import halo_mass_function
import fieldize
import scipy
import scipy.integrate as integ
import scipy.weave

class TotalHaloHI:
    """Find the average HI fraction in a halo
    This is like Figure 9 of Tescari & Viel"""
    def __init__(self,snap_dir,snapnum,minpart=1000):
        #f np > 1.4.0, we have in1d
        if not re.match("1\.[4-9]",np.version.version):
            print "Need numpy 1.4 for in1d: without it this is unfeasibly slow"
        #Get halo catalog
        subs=readsubf.subfind_catalog(snap_dir,snapnum,masstab=True,long_ids=True)
        #Get list of halos resolved with > minpart particles
        ind=np.where(subs.sub_len > minpart)
        self.nHI=np.zeros(np.size(ind))
        self.tot_found=np.zeros(np.size(ind))
        print "Found ",np.size(ind)," halos with > ",minpart,"particles"
        #Get particle ids for each subhalo
        sub_ids=[readsubf.subf_ids(snap_dir,snapnum,np.sum(subs.sub_len[0:i]),subs.sub_len[i],long_ids=True).SubIDs for i in np.ravel(ind)]
        all_sub_ids=np.concatenate(sub_ids)
        print "Got particle id lists"
        #Internal gadget mass unit: 1e10 M_sun in g
        UnitMass_in_g=1.989e43
        UnitLength_in_cm=3.085678e21
        #Now find the average HI for each halo
        for fnum in range(0,500):
            try:
                f=hdfsim.get_file(snapnum,snap_dir,fnum)
            except IOError:
                break
            bar=f["PartType0"]
            iids=np.array(bar["ParticleIDs"],dtype=np.uint64)
            irho=np.array(bar["Density"],dtype=np.float64)*(UnitMass_in_g/UnitLength_in_cm**3)
            inH0 = cold_gas.get_reproc_rhoHI(bar)/irho
            #Find a superset of all the elements
            hind=np.where(np.in1d(iids,all_sub_ids))
            ids=iids[hind]
            nH0=inH0[hind]
            print "File ",fnum," has ",np.size(hind)," halo particles"
            #Assign each subset to the right halo
            tmp=[nH0[np.where(np.in1d(sub,ids))] for sub in sub_ids]
            self.tot_found+=np.array([np.size(i) for i in tmp])
            self.nHI+=np.array([np.sum(i) for i in tmp])
        print "Found ",np.sum(self.tot_found)," gas particles"
        self.nHI/=self.tot_found
        self.mass=subs.sub_mass[ind]
        return

class HaloHI:
    """Class for calculating properties of DLAs in a simulation.
    Stores grids of the neutral hydrogen density around a given halo,
    which are used to derive the halo properties.

    Parameters:
        dir - Simulation directory
        snapnum - Number of simulation
        minpart - Minimum size of halo to consider, in particles
        ngrid - Size of grid to store values on
        maxdist - Maximum extent of grid in kpc.
        self.sub_nHI_grid is a list of neutral hydrogen grids, in log(N_HI / cm^-2) units.
        self.sub_mass is a list of halo masses
        self.sub_cofm is a list of halo positions"""
    def __init__(self,snap_dir,snapnum,minpart=10**4,ngrid=33,maxdist=100.):
        self.minpart=minpart
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        self.ngrid=ngrid
        self.maxdist=maxdist
        #f np > 1.4.0, we have in1d
        if not re.match("1\.[4-9]",np.version.version):
            print "Need numpy 1.4 for in1d: without it this is unfeasibly slow"
        #Get halo catalog
        subs=readsubf.subfind_catalog(self.snap_dir,snapnum,masstab=True,long_ids=True)
        #Get list of halos resolved with > minpart particles
        ind=np.where(subs.sub_len > minpart)
        self.nHI=np.zeros(np.size(ind))
        self.tot_found=np.zeros(np.size(ind))
        self.nhalo=np.size(ind)
        print "Found ",self.nhalo," halos with > ",minpart,"particles"
        #Get particle center of mass 
        self.sub_cofm=np.array(subs.sub_pos[ind])
        #halo masses
        self.sub_mass=np.array(subs.sub_mass[ind])
        del subs
        #Grid to put paticles on
        f=hdfsim.get_file(snapnum,self.snap_dir,0)
        self.redshift=f["Header"].attrs["Redshift"]
        f.close()
        self.sub_nHI_grid=self.set_nHI_grid(ngrid,maxdist)
        return


    def set_nHI_grid(self,ngrid=None,maxdist=None):
        """Set up the grid around each halo where the HI is calculated.
            ngrid - Size of grid to store values on
            maxdist - Maximum extent of grid in kpc.
        """
        if ngrid != None:
            self.ngrid=ngrid
        if maxdist != None:
            self.maxdist=maxdist
        UnitLength_in_cm=3.085678e21
        sub_nHI_grid=[np.zeros((self.ngrid,self.ngrid)) for i in self.sub_cofm]
        #Now grid the HI for each halo
        for fnum in xrange(0,500):
            try:
                f=hdfsim.get_file(self.snapnum,self.snap_dir,fnum)
            except IOError:
                break
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"],dtype=np.float64)
            irhoH0 = cold_gas.get_reproc_rhoHI(bar)
            f.close()
            #Find particles near each halo
            near_halo=[np.where(np.all((np.abs(ipos-sub_pos) < self.maxdist),axis=1)) for sub_pos in self.sub_cofm]
            print "File ",fnum," has ",np.sum([np.size(i) for i in near_halo])," halo particles"
            #positions, centered on each halo, in grid units
            poslist=[ipos[ind] for ind in near_halo]
            coords=[ppos- self.sub_cofm[idx] for idx,ppos in enumerate(poslist)]
            coords=[fieldize.convert_centered(co,self.ngrid,self.maxdist) for co in coords]
            #NH0
            rhoH0 = [irhoH0[ind] for ind in near_halo]
            map(fieldize.ngp, coords,rhoH0,sub_nHI_grid)
        #Linear dimension of each cell in cm
        epsilon=2.*self.maxdist/(self.ngrid)*UnitLength_in_cm
        sub_nHI_grid=[g*epsilon/(1+self.redshift)**2 for g in sub_nHI_grid]
        for ii,grid in enumerate(sub_nHI_grid):
            ind=np.where(grid > 0)
            grid[ind]=np.log(grid[ind])
            sub_nHI_grid[ii]=grid
        return sub_nHI_grid

    def get_sigma_DLA(self):
        """Get the DLA cross-section from the neutral hydrogen column densities found in this class.
        This is defined as the area of all the cells with column density above 10^20.3 cm^-2.
        Returns result in (kpc/h)^2."""
        cell_area=4*self.maxdist**2/self.ngrid**2
        sigma_DLA = [ np.sum(grid[np.where(grid > 20.3)])*cell_area for grid in self.sub_nHI_grid]
        return sigma_DLA

class DNdlaDz:
    """Get the DLA number density as a function of redshift, defined as:
    d N_DLA / dz ( > M, z) = dr/dz int^\infinity_M n_h(M', z) sigma_DLA(M',z) dM'
    where n_h is the Sheth-Torman mass function, and
    sigma_DLA is a power-law fit to self.sigma_DLA.
    Parameters:
        sigma_DLA -  List of DLA cross-sections
        masses - List of DLA masses
        redshift
        Omega_M
        Omega_L"""
    def __init__(self, sigma_DLA,halo_mass, redshift,Omega_M=0.27, Omega_L = 0.73, hubble=0.7):
        self.redshift=redshift
        self.Omega_M = Omega_M
        self.Omega_L = Omega_L
        self.hubble=hubble
        #log of halo mass limits in M_sun
        self.log_mass_lim=(6,20)
        #Fit to the DLA abundance
        logmass=np.log(halo_mass)-12
        logsigma=np.log(sigma_DLA)
        (self.alpha,self.beta)=scipy.polyfit(logmass,logsigma,1)
        #Halo mass function object
        self.halo_mass=halo_mass_function.HaloMassFunction(redshift,omega_m=Omega_M, omega_l=Omega_L, hubble=hubble,log_mass_lim=self.log_mass_lim)

    def sigma_DLA_fit(self,M):
        """Returns sigma_DLA(M) for the linear regression fit"""
        return np.exp(self.alpha*(np.log(M)-12)+self.beta)


    def drdz(self,zz):
        """Calculates dr/dz in a flat cosmology"""
        #Speed of light in cgs units
        light=2.9979e10
        #h * 100 km/s/Mpc in 1/s
        h100=3.2407789e-18*self.hubble
        return light/h100*np.sqrt(self.Omega_M*(1+zz)**3+self.Omega_L)

    def get_N_DLA_dz(self, mass=1e9):
        """Get the DLA number density as a function of redshift, defined as:
        d N_DLA / dz ( > M, z) = dr/dz int^\infinity_M n_h(M', z) sigma_DLA(M',z) dM'
        where n_h is the Sheth-Torman mass function, and
        sigma_DLA is a power-law fit to self.sigma_DLA.
        Parameters:
            lower_mass in M_sun.
        """
        result = integ.quad(self.NDLA_integrand,np.log(mass),np.log(10.**self.log_mass_lim[1]), epsrel=1e-2)
        return self.drdz(self.redshift)*result[0]

    def NDLA_integrand(self,logM):
        """Integrand for above"""
        M=np.exp(logM)
        return self.sigma_DLA_fit(M)*self.halo_mass.dndm(M)*M
