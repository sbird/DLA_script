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
import hsml
import scipy
import scipy.integrate as integ
import scipy.weave

class TotalHaloHI:
    """Find the average HI fraction in a halo
    This is like Figure 9 of Tescari & Viel"""
    def __init__(self,snap_dir,snapnum,minpart=1000,hubble=0.7):
        self.snap_dir=snap_dir
        self.snapnum=snapnum
        #proton mass in g
        self.protonmass=1.66053886e-24
        #Internal gadget mass unit: 1e10 M_sun/h in g/h
        UnitMass_in_g=1.989e43
        #1 M_sun in g
        SolarMass_in_g=1.989e33
        #Internal gadget mass unit: 1 kpc/h in cm/h
        UnitLength_in_cm=3.085678e21
        self.hy_mass = 0.76 # Hydrogen massfrac
        self.minpart=minpart
        self.hubble=hubble
        #Name of savefile
        self.savefile=path.join(self.snap_dir,"snapdir_"+str(self.snapnum).rjust(3,'0'),"tot_hi_grid.npz")
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
            #Initialise arrays
            self.nHI=np.zeros(np.size(ind))
            self.tot_found=np.zeros(np.size(ind))
            #Put masses in M_sun/h
            self.mass=subs.sub_mass[ind]*UnitMass_in_g/SolarMass_in_g
            print "Found ",np.size(ind)," halos with > ",minpart,"particles"
            #Get particle ids for each subhalo
            sub_ids=[readsubf.subf_ids(snap_dir,snapnum,np.sum(subs.sub_len[0:i]),subs.sub_len[i],long_ids=True).SubIDs for i in np.ravel(ind)]
            #NOTE: this will in general be much larger than the number of particles we want to process,
            #because it includes the DM.
            all_sub_ids=np.concatenate(sub_ids)
            del subs
            print "Got particle id lists"
            star=cold_gas.StarFormation(hubble=self.hubble)
            #Now find the average HI for each halo
            for fnum in range(0,500):
                try:
                    f=hdfsim.get_file(snapnum,snap_dir,fnum)
                except IOError:
                    break
                bar=f["PartType0"]
                iids=np.array(bar["ParticleIDs"],dtype=np.uint64)
                #Density in (g/h)/(cm/h)^3 = g/cm^3 h^2
                irho=np.array(bar["Density"],dtype=np.float64)*(UnitMass_in_g/UnitLength_in_cm**3)
                #nH0 in atoms/cm^3 (NOTE NO h!)
                inH0 = star.get_reproc_rhoHI(bar)
                #Convert to neutral fraction: this is in neutral atoms/total hydrogen.
                inH0/=(irho*self.hubble**2*self.hy_mass/self.protonmass)
                #So nH0 is dimensionless
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
            #If a halo has no gas particles
            ind=np.where(self.tot_found > 0)
            self.nHI[ind]/=self.tot_found[ind]
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
        reload_file - Ignore saved files if true
        self.sub_nHI_grid is a list of neutral hydrogen grids, in log(N_HI / cm^-2) units.
        self.sub_mass is a list of halo masses
        self.sub_cofm is a list of halo positions"""
    def __init__(self,snap_dir,snapnum,minpart=10**4,ngrid=None,maxdist=100.,halo_list=None,reload_file=False):
        self.minpart=minpart
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        self.maxdist=maxdist
        #Internal gadget mass unit: 1e10 M_sun/h in g/h
        self.UnitMass_in_g=1.989e43
        #1 M_sun in g
        self.SolarMass_in_g=1.989e33
        #Internal gadget length unit: 1 kpc/h in cm/h
        self.UnitLength_in_cm=3.085678e21
        self.UnitVelocity_in_cm_per_s=1e5
        #Name of savefile
        self.savefile=path.join(self.snap_dir,"snapdir_"+str(self.snapnum).rjust(3,'0'),"halohi_grid.npz")
        #For printing
        self.once=False
        try:
            if reload_file:
                raise KeyError("reloading")
            #First try to load from a file
            grid_file=np.load(self.savefile)

            if  not (grid_file["maxdist"] == self.maxdist and grid_file["minpart"] == self.minpart):
                raise KeyError("File not for this structure")
            #Otherwise...
            self.sub_nHI_grid = grid_file["sub_nHI_grid"]
            self.sub_gas_grid = grid_file["sub_gas_grid"]
            self.sub_mass = grid_file["sub_mass"]
            self.sub_cofm=grid_file["sub_cofm"]
            self.redshift=grid_file["redshift"]
            self.ind=grid_file["halo_ind"]
            self.omegam=grid_file["omegam"]
            self.omegal=grid_file["omegal"]
            self.hubble=grid_file["hubble"]
            self.box=grid_file["box"]
            self.ngrid=grid_file["ngrid"]
            grid_file.close()
            if halo_list != None:
                self.sub_nHI_grid=self.sub_nHI_grid[halo_list]
                self.sub_gas_grid=self.sub_gas_grid[halo_list]
                self.sub_mass=self.sub_mass[halo_list]
                self.sub_cofm=self.sub_cofm[halo_list]
                self.nhalo=np.size(halo_list)
        except (IOError,KeyError):
            #Otherwise regenerate from the raw data
            #Get halo catalog
            subs=readsubf.subfind_catalog(self.snap_dir,snapnum,masstab=True,long_ids=True)
            #Get list of halos resolved with > minpart particles
            ind=np.where(subs.sub_len > minpart)
            #Store the indices of the halos we are using
            self.ind=ind
            self.nhalo=np.size(self.ind)
            print "Found ",self.nhalo," halos with > ",minpart,"particles"
            #Get particle center of mass
            self.sub_cofm=np.array(subs.sub_pos[ind])
            #halo masses in M_sun/h
            self.sub_mass=np.array(subs.sub_mass[ind])*self.UnitMass_in_g/self.SolarMass_in_g
            del subs
            if halo_list != None:
                self.sub_mass=self.sub_mass[halo_list]
                self.sub_cofm=self.sub_cofm[halo_list]
                self.nhalo=np.size(halo_list)
            #Simulation parameters
            f=hdfsim.get_file(snapnum,self.snap_dir,0)
            self.redshift=f["Header"].attrs["Redshift"]
            self.hubble=f["Header"].attrs["HubbleParam"]
            self.box=f["Header"].attrs["BoxSize"]
            npart=f["Header"].attrs["NumPart_Total"]+2**32*f["Header"].attrs["NumPart_Total_HighWord"]
            self.omegam=f["Header"].attrs["Omega0"]
            self.omegal=f["Header"].attrs["OmegaLambda"]
            f.close()
            #Set ngrid to be the gravitational softening length
            if ngrid == None:
                self.ngrid=int(np.ceil(40*npart[1]**(1./3)/self.box*2*self.maxdist))
            else:
                self.ngrid=int(ngrid)
            (self.sub_gas_grid,self.sub_nHI_grid)=self.set_nHI_grid()
        return

    def save_file(self):
        """
        Saves grids to a file, because they are slow to generate.
        File is hard-coded to be $snap_dir/snapdir_$snapnum/hi_grid_$ngrid.npz.
        """
        np.savez_compressed(self.savefile,maxdist=self.maxdist,minpart=self.minpart,ngrid=self.ngrid,sub_mass=self.sub_mass,sub_nHI_grid=self.sub_nHI_grid,sub_gas_grid=self.sub_gas_grid,sub_cofm=self.sub_cofm,redshift=self.redshift,hubble=self.hubble,box=self.box,omegam=self.omegam,omegal=self.omegal,halo_ind=self.ind)



    def set_nHI_grid(self,ngrid=None,maxdist=None):
        """Set up the grid around each halo where the HI is calculated.
            ngrid - Size of grid to store values on
            maxdist - Maximum extent of grid in kpc/h
        Returns:
            sub_nHI_grid - a grid containing the integrated N_HI in neutral atoms/cm^-2
                           summed along the z-axis
        """
        if ngrid != None:
            self.ngrid=ngrid
        if maxdist != None:
            self.maxdist=maxdist
        sub_nHI_grid=np.zeros((self.nhalo,self.ngrid,self.ngrid))
        sub_gas_grid=np.zeros((self.nhalo,self.ngrid,self.ngrid))
        star=cold_gas.StarFormation(hubble=self.hubble)
        self.once=True
        #Now grid the HI for each halo
        for fnum in xrange(0,500):
            try:
                f=hdfsim.get_file(self.snapnum,self.snap_dir,fnum)
            except IOError:
                break
            print "Starting file ",fnum
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"],dtype=np.float64)
            #Returns neutral density in atoms/cm^3
            irhoH0 = star.get_reproc_rhoHI(bar)
            smooth = hsml.get_smooth_length(bar)
            # gas density in g/cm^3
            irho=np.array(bar["Density"],dtype=np.float64)*(self.UnitMass_in_g/self.UnitLength_in_cm**3)*self.hubble**2
            protonmass=1.66053886e-24
            hy_mass = 0.76 # Hydrogen massfrac
            # gas density in hydrogen atoms/cm^3
            irho*=(hy_mass/protonmass)
            f.close()
            #Convert smoothing lengths to grid coordinates.
            smooth*=(self.ngrid/(2*self.maxdist))
            #Perform the grid interpolation
            self.sub_gridize_single_file(ipos,smooth,irho,sub_gas_grid,irhoH0,sub_nHI_grid)
        #Linear dimension of each cell in cm:
        #               kpc/h                   1 cm/kpc
        epsilon=2.*self.maxdist/(self.ngrid)*self.UnitLength_in_cm/self.hubble
        sub_nHI_grid*=(epsilon/(1+self.redshift)**2)
        sub_gas_grid*=(epsilon/(1+self.redshift)**2)
        ind = np.where(sub_gas_grid > 0)
        sub_gas_grid[ind] = np.log10(sub_gas_grid[ind])
        ind = np.where(sub_nHI_grid > 0)
        sub_nHI_grid[ind] = np.log10(sub_nHI_grid[ind])
        return (sub_gas_grid,sub_nHI_grid)

    def sub_gridize_single_file(self,ipos,ismooth,irho,sub_gas_grid,irhoH0,sub_nHI_grid):
        """Helper function for sub_gas_grid and sub_nHI_grid
            that puts data arrays loaded from a particular file onto the grid.
            Arguments:
                pos - Position array
                rho - Density array to be interpolated
                smooth - Smoothing lengths
                sub_grid - Grid to add the interpolated data to
        """
        #Find particles near each halo
        for ii in range(0,self.nhalo):
            sub_pos=self.sub_cofm[ii]
            indx=np.where(np.abs(ipos[:,0]-sub_pos[0]) < self.maxdist)
            pposx=ipos[indx]
            indz=np.where(np.all(np.abs(pposx[:,1:3]-sub_pos[1:3]) < self.maxdist,axis=1))
            if np.size(indz) == 0:
                continue
            #coords in grid units
            coords=fieldize.convert_centered(pposx[indz]-sub_pos,self.ngrid,2*self.maxdist)
            #NH0
            smooth = (ismooth[indx])[indz]
            if self.once:
                print "Av. smoothing length is ",np.mean(smooth)*2*self.maxdist/self.ngrid," kpc/h ",np.mean(smooth), "grid cells"
                self.once=False
            rho=(irho[indx])[indz]
            fieldize.cic_str(coords,rho,sub_gas_grid[ii,:,:],smooth)
            rhoH0=(irhoH0[indx])[indz]
            fieldize.cic_str(coords,rhoH0,sub_nHI_grid[ii,:,:],smooth)
        return

    def get_sigma_DLA(self,DLA_cut=20.3):
        """Get the DLA cross-section from the neutral hydrogen column densities found in this class.
        This is defined as the area of all the cells with column density above 10^DLA_cut (10^20.3) cm^-2.
        Returns result in (kpc/h)^2."""
        cell_area=(2.*self.maxdist/self.ngrid)**2
        sigma_DLA = np.array([ np.shape(np.where(grid > DLA_cut))[1]*cell_area for grid in self.sub_nHI_grid])
        return sigma_DLA

    def sigma_DLA_fit(self,M,DLA_cut=20.3):
        """Returns sigma_DLA(M) for the linear regression fit"""
        #Fit to the DLA abundance
        s_DLA=self.get_sigma_DLA(DLA_cut)
        ind=np.where((s_DLA > 0.))
        logmass=np.log(self.sub_mass[ind])-12
        logsigma=np.log(s_DLA[ind])
        if np.size(logsigma) == 0:
            (self.alpha,self.beta)=(0,0)
        else:
            (self.alpha,self.beta)=scipy.polyfit(logmass,logsigma,1)
        return np.exp(self.alpha*(np.log(M)-12)+self.beta)


    def get_absorber_area(self,minN,maxN):
        """Return the total area (in kpc/h^2) covered by absorbers with column density covered by a given bin"""
        #Number of grid cells
        flat_grid=np.ravel(self.sub_nHI_grid)
        cells=np.shape(np.where(np.logical_and(flat_grid > minN,flat_grid < maxN)))[1]
        #Area of grid cells in kpc/h^2
        cell_area=(1./self.ngrid)**2
        return cells*cell_area

    def get_radial_profile(self,halo,minR,maxR):
        """Returns the nHI density summed radially
           (but really in Cartesian coordinates).
           So returns R_HI (cm^-1).
           Should use bins in r significantly larger
           than the grid size.
        """
        #This is an integral over an annulus in Cartesians
        #Find r in grid units:
        total=0
        gminR=minR/(2.*self.maxdist)*self.ngrid
        gmaxR=maxR/(2.*self.maxdist)*self.ngrid
        cen=self.ngrid/2.
        #Broken part of the annulus:
        for x in xrange(-gminR,gminR):
            miny=np.sqrt(gminR**2-x**2)+cen
            maxy=np.sqrt(gmaxR**2-x**2)+cen
            total+=np.sum(self.sub_nHI_grid[halo,x+self.ngrid/2,miny:maxy],axis=2)
            total+=np.sum(self.sub_nHI_grid[halo,x+self.ngrid/2,-maxy:-miny],axis=2)
        #Complete part of annulus
        for x in xrange(gminR,gmaxR):
            maxy=np.sqrt(gmaxR**2-x**2)+cen
            maxy=-np.sqrt(gmaxR**2-x**2)+cen
            total+=np.sum(self.sub_nHI_grid[halo,x+cen,miny:maxy],axis=2)
            total+=np.sum(self.sub_nHI_grid[halo,-x+cen,miny:maxy],axis=2)
        return total*(2.*self.maxdist)/self.ngrid



    def absorption_distance(self):
        """Compute X(z), the absorption distance per sightline (eq. 9 of Nagamine et al 2003)
        in dimensionless units."""
        #h * 100 km/s/Mpc in h/s
        h100=3.2407789e-18
        # in cm/s
        light=2.9979e10
        #Units: h/s   s/cm                        kpc/h      cm/kpc
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
        #Grid size (in cm^2)
        cell_area=(1./self.ngrid)**2
        dX=self.absorption_distance()
        flat_grid=np.ravel(self.sub_nHI_grid)
        (f_N,NHI_table)=np.histogram(flat_grid,np.log10(NHI_table))
        NHI_table=(10.)**NHI_table
        #To compensate for any rounding
        dlogN_real = (np.log10(NHI_table[-1])-np.log10(NHI_table[0]))/(np.size(NHI_table)-1)
        f_N=(1.*f_N*cell_area)/((NHI_table[0:-1]*(10.)**dlogN_real-NHI_table[0:-1])*dX)
        return (NHI_table[0:-1]*(10.)**(dlogN_real/2.), f_N)

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

    def get_dndm(self,minM,maxM):
        """Get the halo mass function from the simulations,
        in units of h^4 M_sun^-1 Mpc^-3.
        Parameters:
            minM and maxM are the sides of the bin to use.
        """
        #Number of halos in this mass bin in the whole box
        Nhalo=np.shape(np.where((self.sub_mass <= maxM)*(self.sub_mass > minM)))[1]
        Mpch_in_cm=3.085678e24
        #Convert to halos per Mpc/h^3
        Nhalo/=(self.box*self.UnitLength_in_cm/Mpch_in_cm)**3
        #Convert to per unit mass
        return Nhalo/(maxM-minM)



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
        #Make it an array, not a list
        s_DLA=np.array(sigma_DLA)
        h_mass=np.array(halo_mass)
        #Fit to the DLA abundance
        ind=np.where((s_DLA > 0.))
        logmass=np.log(h_mass[ind])-12
        logsigma=np.log(s_DLA[ind])
        if np.size(logsigma) == 0:
            (self.alpha,self.beta)=(0,0)
        else:
            (self.alpha,self.beta)=scipy.polyfit(logmass,logsigma,1)
        #Halo mass function object
        self.halo_mass=halo_mass_function.HaloMassFunction(redshift,omega_m=Omega_M, omega_l=Omega_L, hubble=hubble,log_mass_lim=self.log_mass_lim)

    def sigma_DLA_fit(self,M):
        """Returns sigma_DLA(M) for the linear regression fit"""
        return np.exp(self.alpha*(np.log(M)-12)+self.beta)


    def drdz(self,zz):
        """Calculates dr/dz in a flat cosmology in units of cm/h"""
        #Speed of light in cm/s
        light=2.9979e10
        #h * 100 km/s/Mpc in h/s
        h100=3.2407789e-18
        #       cm/s   s/h   =>
        return light/h100*np.sqrt(self.Omega_M*(1+zz)**3+self.Omega_L)

    def get_N_DLA_dz(self, mass=1e9):
        """Get the DLA number density as a function of redshift, defined as:
        d N_DLA / dz ( > M, z) = dr/dz int^\infinity_M n_h(M', z) sigma_DLA(M',z) dM'
        where n_h is the Sheth-Torman mass function, and
        sigma_DLA is a power-law fit to self.sigma_DLA.
        Parameters:
            lower_mass in M_sun/h.
        """
        result = integ.quad(self.NDLA_integrand,np.log10(mass),self.log_mass_lim[1], epsrel=1e-2)
        #drdz is in cm/h, while the rest is in kpc/h, so convert.
        return self.drdz(self.redshift)*result[0]/3.085678e21

    def NDLA_integrand(self,log10M):
        """Integrand for above"""
        M=10**log10M
        #sigma_DLA is in kpc/h^2, while halo_mass is in h^4 M_sun^-1 Mpc^(-3), so convert.
        return self.sigma_DLA_fit(M)*self.halo_mass.dndm(M)*M/(10**9)

