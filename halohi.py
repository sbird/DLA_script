# vim: set fileencoding=utf-8
"""Module for creating the DLA hydrogen density plots, as found in Tescari & Viel,
and Nagamine, Springel and Hernquist, 2003.

Classes:
    TotalHaloHI - Finds the average HI fraction in a halo
    HaloHI - Creates a grid around the halo center with the HI fraction calculated at each grid cell
"""
import numpy as np
import readsubf
import hdfsim
import math
import os.path as path
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
        self.snap_dir=snap_dir
        self.snapnum=snapnum
        #proton mass in g
        self.protonmass=1.66053886e-24
        self.hy_mass = 0.76 # Hydrogen massfrac
        self.minpart=minpart
        #Name of savefile
        self.savefile=path.join(self.snap_dir,"snapdir_"+str(self.snapnum),"tot_hi_grid.npz")
        try:
            #First try to load from a file
            grid_file=np.load(self.savefile)

            if  grid_file["minpart"] != self.minpart:
                raise KeyError("File not for this structure")
            #Otherwise...
            self.nHI = grid_file["nHI"]
            self.mass = grid_file["mass"]
            self.tot_found=grid_file["tot_found"]
            grid_file.close()
        except (IOError,KeyError):
            #Get halo catalog
            subs=readsubf.subfind_catalog(snap_dir,snapnum,masstab=True,long_ids=True)
            #Get list of halos resolved with > minpart particles
            ind=np.where(subs.sub_len > minpart)
            #Initialise arays
            self.nHI=np.zeros(np.size(ind))
            self.tot_found=np.zeros(np.size(ind))
            #Put masses in M_sun
            self.mass=subs.sub_mass[ind]*1e10
            print "Found ",np.size(ind)," halos with > ",minpart,"particles"
            #Get particle ids for each subhalo
            sub_ids=[readsubf.subf_ids(snap_dir,snapnum,np.sum(subs.sub_len[0:i]),subs.sub_len[i],long_ids=True).SubIDs for i in np.ravel(ind)]
            all_sub_ids=np.concatenate(sub_ids)
            del subs
            print "Got particle id lists"
            #Internal gadget mass unit: 1e10 M_sun in g
            UnitMass_in_g=1.989e43
            UnitLength_in_cm=3.085678e21
            star=cold_gas.StarFormation()
            #Now find the average HI for each halo
            for fnum in range(0,500):
                try:
                    f=hdfsim.get_file(snapnum,snap_dir,fnum)
                except IOError:
                    break
                bar=f["PartType0"]
                iids=np.array(bar["ParticleIDs"],dtype=np.uint64)
                irho=np.array(bar["Density"],dtype=np.float64)*(UnitMass_in_g/UnitLength_in_cm**3)
                inH0 = star.get_reproc_rhoHI(bar)/(irho*self.hy_mass/self.protonmass)
                #Find a superset of all the elements
                hind=np.where(np.in1d(iids,all_sub_ids))
                ids=iids[hind]
                nH0=inH0[hind]
                print "File ",fnum," has ",np.size(hind)," halo particles"
                #Assign each subset to the right halo
                tmp=[nH0[np.where(np.in1d(ids,sub))] for sub in sub_ids]
                self.tot_found+=np.array([np.size(i) for i in tmp])
                self.nHI+=np.array([np.sum(i) for i in tmp])
            print "Found ",np.sum(self.tot_found)," gas particles"
            self.nHI/=self.tot_found
        return

    def save_file(self):
        """
        Saves grids to a file, because they are slow to generate.
        File is hard-coded to be $snap_dir/snapdir_$snapnum/tot_hi_grid.npz.
        """
        np.savez_compressed(self.savefile,minpart=self.minpart,mass=self.mass,nHI=self.nHI,tot_found=self.tot_found)



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
        halo_list - If not None, only consider halos in the list
        slice - Only consider particles near the z-axis if true
        reload - Ignore saved files if true
        self.sub_nHI_grid is a list of neutral hydrogen grids, in log(N_HI / cm^-2) units.
        self.sub_mass is a list of halo masses
        self.sub_cofm is a list of halo positions"""
    def __init__(self,snap_dir,snapnum,minpart=10**4,ngrid=33,maxdist=100.,halo_list=None,slice=False,reload=False):
        self.minpart=minpart
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        self.ngrid=ngrid
        self.maxdist=maxdist
        self.slice=slice
        #Internal gadget mass unit: 1e10 M_sun in g
        self.UnitMass_in_g=1.989e43
        #Internal gadget length unit: 1 kpc in cm
        self.UnitLength_in_cm=3.085678e21
        self.UnitVelocity_in_cm_per_s=1e5
        #Name of savefile
        self.savefile=path.join(self.snap_dir,"snapdir_"+str(self.snapnum),"hi_grid_"+str(ngrid)+".npz")
        try:
            if reload:
                raise KeyError("reloading")
            #First try to load from a file
            grid_file=np.load(self.savefile)

            if  not (grid_file["maxdist"] == self.maxdist and grid_file["minpart"] == self.minpart and self.ngrid == grid_file["ngrid"]):
                raise KeyError("File not for this structure")
            #Otherwise...
            self.sub_nHI_grid = grid_file["sub_nHI_grid"]
            self.sub_mass = grid_file["sub_mass"]
            self.sub_cofm=grid_file["sub_cofm"]
            self.redshift=grid_file["redshift"]
            self.omegam=grid_file["omegam"]
            self.omegal=grid_file["omegal"]
            self.hubble=grid_file["hubble"]
            self.box=grid_file["box"]
            grid_file.close()
            if halo_list != None:
                self.sub_nHI_grid=self.sub_nHI_grid[halo_list]
                self.sub_mass=self.sub_mass[halo_list]
                self.sub_cofm=self.sub_cofm[halo_list]
        except (IOError,KeyError):
            #Otherwise regenerate from the raw data
            #Get halo catalog
            subs=readsubf.subfind_catalog(self.snap_dir,snapnum,masstab=True,long_ids=True)
            #Get list of halos resolved with > minpart particles
            ind=np.where(subs.sub_len > minpart)
            self.nhalo=np.size(ind)
            print "Found ",self.nhalo," halos with > ",minpart,"particles"
            #Get particle center of mass
            self.sub_cofm=np.array(subs.sub_pos[ind])
            #halo masses in M_sun
            self.sub_mass=np.array(subs.sub_mass[ind])*self.UnitMass_in_g/1.989e33
            del subs
            if halo_list != None:
                self.sub_mass=self.sub_mass[halo_list]
                self.sub_cofm=self.sub_cofm[halo_list]
            #Simulation parameters
            f=hdfsim.get_file(snapnum,self.snap_dir,0)
            self.redshift=f["Header"].attrs["Redshift"]
            self.hubble=f["Header"].attrs["HubbleParam"]
            self.box=f["Header"].attrs["BoxSize"]
            self.omegam=f["Header"].attrs["Omega0"]
            self.omegal=f["Header"].attrs["OmegaLambda"]
            f.close()
            self.sub_nHI_grid=self.set_nHI_grid(ngrid,maxdist)
        return

    def save_file(self):
        """
        Saves grids to a file, because they are slow to generate.
        File is hard-coded to be $snap_dir/snapdir_$snapnum/hi_grid_$ngrid.npz.
        """
        np.savez_compressed(self.savefile,maxdist=self.maxdist,minpart=self.minpart,ngrid=self.ngrid,sub_mass=self.sub_mass,sub_nHI_grid=self.sub_nHI_grid,sub_cofm=self.sub_cofm,redshift=self.redshift,hubble=self.hubble,box=self.box,omegam=self.omegam,omegal=self.omegal)



    def set_nHI_grid(self,ngrid=None,maxdist=None):
        """Set up the grid around each halo where the HI is calculated.
            ngrid - Size of grid to store values on
            maxdist - Maximum extent of grid in kpc.
        """
        if ngrid != None:
            self.ngrid=ngrid
        if maxdist != None:
            self.maxdist=maxdist
        sub_nHI_grid=[np.zeros((self.ngrid,self.ngrid)) for i in self.sub_cofm]
        star=cold_gas.StarFormation()
        #Now grid the HI for each halo
        for fnum in xrange(0,500):
            try:
                f=hdfsim.get_file(self.snapnum,self.snap_dir,fnum)
            except IOError:
                break
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"],dtype=np.float64)
            irhoH0 = star.get_reproc_rhoHI(bar)
            f.close()
            #Find particles near each halo
            if not self.slice:
                near_halo=[np.where(np.all((np.abs(ipos-sub_pos) < self.maxdist),axis=1)) for sub_pos in self.sub_cofm]
            else:
                #Only consider particles near z=0
                near_halo=[np.where(np.all((np.abs(ipos-sub_pos) < self.maxdist),axis=1)*(np.abs(ipos[:,2]-sub_pos[2])< self.maxdist/self.ngrid) ) for sub_pos in self.sub_cofm]
            print "File ",fnum," has ",np.sum([np.size(i) for i in near_halo])," halo particles"
            #positions, centered on each halo, in grid units
            poslist=[ipos[ind] for ind in near_halo]
            coords=[ppos- self.sub_cofm[idx] for idx,ppos in enumerate(poslist)]
            coords=[fieldize.convert_centered(co,self.ngrid,2*self.maxdist) for co in coords]
            #NH0
            rhoH0 = [irhoH0[ind] for ind in near_halo]
            map(fieldize.tsc, coords,rhoH0,sub_nHI_grid)
        #Linear dimension of each cell in cm
        epsilon=2.*self.maxdist/(self.ngrid)*self.UnitLength_in_cm
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
        cell_area=(2.*self.maxdist/self.ngrid)**2
        sigma_DLA = [ np.shape(np.where(grid > 20.3))[1]*cell_area for grid in self.sub_nHI_grid]
        return sigma_DLA

    def get_absorber_area(self,minN,maxN):
        """Return the total area (in kpc^2) covered by absorbers with column density covered by a given bin"""
        #Number of grid cells
        cells=np.sum([np.shape(np.where((grid > minN)* (grid < maxN)))[1] for grid in self.sub_nHI_grid])
        #Area of grid cells
        cell_area=(2.*self.maxdist/self.ngrid)**2
        return cells*cell_area

    def absorption_distance(self):
        """Compute X(z), the absorption distance per sightline (eq. 9 of Nagamine et al 2003)"""
        #h * 100 km/s/Mpc in 1/s
        h100=3.2407789e-18*self.hubble
        light=2.9979e10
        return h100/light*(1+self.redshift)**2*self.box*self.UnitLength_in_cm

    def column_density_function(self,dlogN=0.2, minN=20.3, maxN=30.):
        """
        This computes the DLA column density function, which is the number
        of absorbers per sight line with HI column densities in the interval
        [NHI, NHI+dNHI] at the absorption distance X.
        Absorption distance is simply a single simulation box.
        A sightline is assumed to be equivalent to one grid cell.
        That is, there is presumed to be only one halo in along the sightline
        encountering a given halo.

        So we have f(N) = d n_DLA/ dN dX
        and n_DLA(N) = number of absorbers in this column density bin.
                     = fraction of total (grid? box?) area covered by this column density bin
        ie, f(N) = n_DLA / ΔN / ΔX
        Note f(N) has dimensions of cm^2, because N has units of cm^-2 and X is dimensionless.

        Parameters:
            dlogN - bin spacing to aim for (may not actually be reached)
            minN - minimum log N
            maxN - maximum log N

        Returns:
            (NHI, f_N_table) - N_HI (binned in log) and corresponding f(N)
        """
        NHI_table = np.logspace(minN, maxN,(maxN-minN)/dlogN,endpoint=True)
        f_N = np.empty(np.size(NHI_table))
        #To compensate for any rounding
        dlogN_real = (np.log10(NHI_table[-1])-np.log10(NHI_table[0]))/(np.size(NHI_table)-1)
        #Grid size (in cm^2)
        dX=self.absorption_distance()
        for ii,N in enumerate(NHI_table):
            logN=np.log10(N)
            n_DLA_N = self.get_absorber_area(logN,logN+dlogN_real)/self.box**2
            f_N[ii] = n_DLA_N/dlogN_real/N/dX

        return (NHI_table, f_N)

    def omega_DLA(self, maxN):
        """Compute Omega_DLA, defined as:
            Ω_DLA = m_p H_0/(c f_HI rho_c) \int_10^20.3^Nmax  f(N,X) N dN
        """
        (NHI_table, f_N) = self.column_density_function(minN=20.3,maxN=maxN)
        dlogN_real = (np.log10(NHI_table[-1])-np.log10(NHI_table[0]))/(np.size(NHI_table)-1)
        omega_DLA=np.sum(NHI_table*f_N*10**dlogN_real)
        h100=3.2407789e-18*self.hubble
        light=2.9979e10
        rho_crit=3*h100**2/(8*math.pi*6.672e-8)
        protonmass=1.66053886e-24
        hy_mass = 0.76 # Hydrogen massfrac
        omega_DLA*=(h100/light)*(protonmass/hy_mass)/rho_crit
        return omega_DLA


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
        self.log_mass_lim=(7,15)
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

