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
import h5py
import math
import os.path as path
import cold_gas
import halo_mass_function
import fieldize
import hsml
import scipy
import scipy.integrate as integ
import scipy.weave

def is_masked(halo,sub_mass,sub_cofm, sub_radii):
    """Find out whether a halo is a mere satellite and if so mask it"""
    near=np.where(np.all((np.abs(sub_cofm[:,:]-sub_cofm[halo,:]) < sub_radii[halo]),axis=1))
    #If there is a larger halo nearby, mask this halo
    return np.size(np.where(sub_mass[near] > sub_mass[halo])) == 0

class TotalHaloHI:
    """Find the average HI fraction in a halo
    This is like Figure 9 of Tescari & Viel"""
    def __init__(self,snap_dir,snapnum,minpart=400):
        self.snap_dir=snap_dir
        self.snapnum=snapnum
        #proton mass in g
        self.protonmass=1.66053886e-24
        #Internal gadget mass unit: 1e10 M_sun/h in g/h
        self.UnitMass_in_g=1.989e43
        #1 M_sun in g
        self.SolarMass_in_g=1.989e33
        #Internal gadget mass unit: 1 kpc/h in cm/h
        self.UnitLength_in_cm=3.085678e21
        self.hy_mass = 0.76 # Hydrogen massfrac
        self.minpart=minpart
        #Name of savefile
        self.savefile=path.join(self.snap_dir,"snapdir_"+str(self.snapnum).rjust(3,'0'),"tot_hi_grid.npz")
        try:
            #First try to load from a file
            grid_file=np.load(self.savefile)

            if  grid_file["minpart"] != self.minpart:
                raise KeyError("File not for this structure")
            #Otherwise...
            self.nHI = grid_file["nHI"]
            self.MHI = grid_file["MHI"]
            self.mass = grid_file["mass"]
            self.hubble = grid_file["hubble"]
            self.box = grid_file["box"]
            self.npart=grid_file["npart"]
            self.omegam = grid_file["omegam"]
            self.Mgas = grid_file["Mgas"]
            self.sub_radii = grid_file["sub_radii"]
            self.tot_found=grid_file["tot_found"]
            self.ind = grid_file["ind"]
            self.cofm=grid_file["cofm"]
            grid_file.close()
        except (IOError,KeyError):
            f=hdfsim.get_file(snapnum,self.snap_dir,0)
            self.hubble=f["Header"].attrs["HubbleParam"]
            self.box=f["Header"].attrs["BoxSize"]
            self.npart=f["Header"].attrs["NumPart_Total"]+2**32*f["Header"].attrs["NumPart_Total_HighWord"]
            self.omegam=f["Header"].attrs["Omega0"]
            f.close()
            #Get halo catalog
            (self.ind,self.mass,self.cofm,self.sub_radii)=self.find_wanted_halos()
            #Initialise arrays
            self.nHI=np.zeros(np.size(self.ind))
            self.MHI=np.zeros(np.size(self.ind))
            self.Mgas=np.zeros(np.size(self.ind))
            self.tot_found=np.zeros(np.size(self.ind))
            print "Found ",np.size(self.ind)," halos with > ",minpart,"particles"
            star=cold_gas.StarFormation(hubble=self.hubble)
            #Now find the average HI for each halo
            for fnum in range(0,500):
                try:
                    f=hdfsim.get_file(snapnum,snap_dir,fnum)
                except IOError:
                    break
                bar=f["PartType0"]
                print "Starting file ",fnum
                #Density in (g/h)/(cm/h)^3 = g/cm^3 h^2
                irho=np.array(bar["Density"],dtype=np.float64)*(self.UnitMass_in_g/self.UnitLength_in_cm**3)
                #nH0 in atoms/cm^3 (NOTE NO h!)
                inH0 = star.get_reproc_rhoHI(bar)
                #Convert to neutral fraction: this is in neutral atoms/total hydrogen.
                inH0/=(irho*self.hubble**2*self.hy_mass/self.protonmass)
                #So nH0 is dimensionless
                #Mass in solar masses
                imass=np.array(bar["Masses"])*self.UnitMass_in_g/self.SolarMass_in_g
                #Positions in kpc/h
                ipos=np.array(bar["Coordinates"],dtype=np.float64)
                #Assign each subset to the right halo
                [self.get_single_file_by_virial(ii,ipos,inH0,imass) for ii in xrange(0,np.size(self.mass))]
            print "Found ",np.sum(self.tot_found)," gas particles"
            #If a halo has no gas particles
            ind=np.where(self.tot_found > 0)
            self.nHI[ind]/=self.tot_found[ind]
        return

    def find_wanted_halos(self):
        """When handed a halo catalogue, remove from it the halos that are close to other, larger halos.
        Select halos via their M_200 mass, defined in terms of the critical density."""
        #Array to note the halos we don't want
        #Get halo catalog
        subs=readsubf.subfind_catalog(self.snap_dir,self.snapnum,masstab=True,long_ids=True)
        #This is rho_c in units of h^-1 1e10 M_sun (kpc/h)^-3
        rhom = 2.78e+11* self.omegam / 1e10 / (1e3**3)
        #Mass of an SPH particle, in units of 1e10 M_sun, x omega_m/ omega_b.
        target_mass = self.box**3 * rhom / self.npart[0]
        #Get list of halos resolved, using a mass cut; cuts off at about 2e9 for 512**3 particles.
        ind=np.where(subs.group_m_crit200 > self.minpart*target_mass)
        #Store the indices of the halos we are using
        #Get particle center of mass, use group catalogue.
        sub_cofm=np.array(subs.group_pos[ind])
        #halo masses in M_sun/h: use M_200
        sub_mass=np.array(subs.group_m_crit200[ind])*self.UnitMass_in_g/self.SolarMass_in_g
        #r200 in kpc.
        sub_radii = np.array(subs.group_r_crit200[ind])
        #Gas mass in M_sun/h
#         sub_gas_mass=np.array(subs.sub_masstab[ind][:,0])*self.UnitMass_in_g/self.SolarMass_in_g
        del subs
        #For each halo
        ind2=np.where([is_masked(ii,sub_mass,sub_cofm,sub_radii) for ii in xrange(0,np.size(sub_mass))])
        ind=(np.ravel(ind)[ind2],)
        sub_mass=sub_mass[ind2]
        sub_cofm=sub_cofm[ind2]
        sub_radii=sub_radii[ind2]
#         sub_gas_mass=sub_gas_mass[ind2]
        return (ind, sub_mass,sub_cofm,sub_radii)


#     def get_virial_radius(self,mass):
#         """Get the virial radius from a virial mass"""
#         #Virial overdensity
#         virden=200.
#         #in cgs
#         G=6.67259e-8
#         hubble=3.2407789e-18  #in h/sec
#         #rho_c in g/cm^3
#         rho_c = 3*(hubble*self.hubble)**2/8/math.pi/G
#         #rho_c in M_sun  / kpc^3 h^2
#         rho_c*=(self.UnitLength_in_cm**3/self.SolarMass_in_g/self.hubble**2)
#         #Now want radius enclosing mass at avg density of virden*rho_c
#         volume=mass/(rho_c*virden)
#         radius = (volume*3./math.pi/4.)**(1./3)
#         return radius

    def get_single_file_by_virial(self,ii,ipos,inH0,imass):
        """Find all particles within the virial radius of the halo, then sum them"""
        #Linear dimension of each cell in cm:
        #               kpc/h                   1 cm/kpc
        #Find particles near each halo
        sub_pos=self.cofm[ii]
        indx=np.where(np.abs(ipos[:,0]-sub_pos[0]) < self.sub_radii[ii])
        pposx=ipos[indx]
        ind=np.where(np.sum((pposx[:,:]-sub_pos[:])**2,axis=1) < self.sub_radii[ii]**2)
        if np.size(ind) == 0:
            return
        #Find nHI and friends
        self.tot_found[ii]+=np.size(ind)
        nH0=(inH0[indx])[ind]
        mass=(imass[indx])[ind]
        self.nHI[ii]+=np.sum(nH0)
        self.MHI[ii]+=np.sum(nH0*mass)
        self.Mgas[ii]+=np.sum(mass)
        return

    def _get_quant(self,ii,sub,nH0,mass,ids):
        """Helper function; takes a halo and gets nHI,
        total HI mass and total particle and mass number."""
        ind=np.where(np.in1d(ids,sub))
        self.tot_found[ii]+=np.size(ind)
        self.nHI[ii]+=np.sum(nH0[ind])
        self.MHI[ii]+=np.sum(nH0[ind]*mass[ind])
        self.Mgas[ii]+=np.sum(mass[ind])
        return

    def get_hi_mass(self,dm_mass):
        """Get the mass of neutral hydrogen in a halo"""
        ind=np.where(np.abs(self.mass/dm_mass -1. ) < 0.01)
        if np.size(ind) == 0:
            return -1
        else:
            return np.ravel(self.MHI[ind])[0]

    def get_gas_mass(self,dm_mass):
        """Get the mass of neutral hydrogen in a halo"""
        ind=np.where(np.abs(self.mass/dm_mass -1. ) < 0.01)
        if np.size(ind) == 0:
            return -1
        else:
            return np.ravel(self.Mgas[ind])[0]

    def save_file(self):
        """
        Saves grids to a file, because they are slow to generate.
        File is hard-coded to be $snap_dir/snapdir_$snapnum/tot_hi_grid.npz.
        """
        np.savez(self.savefile,minpart=self.minpart,mass=self.mass,nHI=self.nHI,tot_found=self.tot_found,MHI=self.MHI,Mgas=self.Mgas,sub_radii=self.sub_radii,ind=self.ind,cofm=self.cofm,hubble=self.hubble,omegam=self.omegam,box=self.box,npart=self.npart)


class HaloHI:
    """Class for calculating properties of DLAs in a simulation.
    Stores grids of the neutral hydrogen density around a given halo,
    which are used to derive the halo properties.

    Parameters:
        dir - Simulation directory
        snapnum - Number of simulation
        minpart - Minimum size of halo to consider, in DM particle masses
        halo_list - If not None, only consider halos in the list
        reload_file - Ignore saved files if true
        self.sub_nHI_grid is a list of neutral hydrogen grids, in log(N_HI / cm^-2) units.
        self.sub_mass is a list of halo masses
        self.sub_cofm is a list of halo positions"""
    def __init__(self,snap_dir,snapnum,minpart=400,reload_file=False,skip_grid=None,savefile=None):
        self.minpart=minpart
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        #Internal gadget mass unit: 1e10 M_sun/h in g/h
        self.UnitMass_in_g=1.989e43
        #1 M_sun in g
        self.SolarMass_in_g=1.989e33
        #Internal gadget length unit: 1 kpc/h in cm/h
        self.UnitLength_in_cm=3.085678e21
        self.UnitVelocity_in_cm_per_s=1e5
        #Name of savefile
        if savefile == None:
            self.savefile=path.join(self.snap_dir,"snapdir_"+str(self.snapnum).rjust(3,'0'),"halohi_grid.hdf5")
        else:
            self.savefile=savefile
        #For printing
        self.once=False
        try:
            if reload_file:
                raise KeyError("reloading")
            #First try to load from a file
            f=h5py.File(self.savefile,'r')
            grid_file=f["HaloData"]
            if  not (grid_file.attrs["minpart"] == self.minpart):
                raise KeyError("File not for this structure")
            #Otherwise...
            self.redshift=grid_file.attrs["redshift"]
            self.omegam=grid_file.attrs["omegam"]
            self.omegal=grid_file.attrs["omegal"]
            self.hubble=grid_file.attrs["hubble"]
            self.box=grid_file.attrs["box"]
            self.npart=grid_file.attrs["npart"]
            self.ngrid = np.array(grid_file["ngrid"])
            self.sub_mass = np.array(grid_file["sub_mass"])
            self.sub_cofm=np.array(grid_file["sub_cofm"])
            self.sub_radii=np.array(grid_file["sub_radii"])
#             self.sub_gas_mass=np.array(grid_file["sub_gas_mass"])
            self.ind=np.array(grid_file["halo_ind"])
            self.nhalo=np.size(self.ind)
            if not skip_grid == 1:
                self.sub_nHI_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
                grp = f["GridHIData"]
                [ grp[str(i)].read_direct(self.sub_nHI_grid[i]) for i in xrange(0,self.nhalo)]
            if not skip_grid == 2:
                self.sub_gas_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
                grp = f["GridGasData"]
                [ grp[str(i)].read_direct(self.sub_gas_grid[i]) for i in xrange(0,self.nhalo)]
            f.close()
            del grid_file
            del f
        except (IOError,KeyError):
            #Otherwise regenerate from the raw data
            #Simulation parameters
            f=hdfsim.get_file(snapnum,self.snap_dir,0)
            self.redshift=f["Header"].attrs["Redshift"]
            self.hubble=f["Header"].attrs["HubbleParam"]
            self.box=f["Header"].attrs["BoxSize"]
            self.npart=f["Header"].attrs["NumPart_Total"]+2**32*f["Header"].attrs["NumPart_Total_HighWord"]
            self.omegam=f["Header"].attrs["Omega0"]
            self.omegal=f["Header"].attrs["OmegaLambda"]
            f.close()
            (self.ind,self.sub_mass,self.sub_cofm,self.sub_radii)=self.find_wanted_halos()
            self.nhalo=np.size(self.ind)
            if minpart == -1:
                #global grid
                self.nhalo = 1
                self.sub_radii=self.box/2.
            else:
                print "Found ",self.nhalo," halos with > ",minpart,"particles"
            #Set ngrid to be the gravitational softening length
            self.ngrid=np.array([int(np.ceil(40*self.npart[1]**(1./3)/self.box*2*rr)) for rr in self.sub_radii])
            self.sub_nHI_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
            self.sub_gas_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
            self.set_nHI_grid()
        return

    def save_file(self):
        """
        Saves grids to a file, because they are slow to generate.
        File is hard-coded to be $snap_dir/snapdir_$snapnum/halohi_grid.hdf5.
        """
        f=h5py.File(self.savefile,'w')
        grp = f.create_group("HaloData")
        grp.attrs["minpart"]=self.minpart
        grp.attrs["redshift"]=self.redshift
        grp.attrs["hubble"]=self.hubble
        grp.attrs["box"]=self.box
        grp.attrs["npart"]=self.npart
        grp.attrs["omegam"]=self.omegam
        grp.attrs["omegal"]=self.omegal
        grp.create_dataset("ngrid",data=self.ngrid)
        grp.create_dataset('sub_mass',data=self.sub_mass)
#         grp.create_dataset('sub_gas_mass',data=self.sub_gas_mass)
        grp.create_dataset('sub_cofm',data=self.sub_cofm)
        grp.create_dataset('sub_radii',data=self.sub_radii)
        grp.create_dataset('halo_ind',data=self.ind)
        grp_grid = f.create_group("GridHIData")
        grp_gas_grid = f.create_group("GridGasData")
        for i in xrange(0,self.nhalo):
            grp_grid.create_dataset(str(i),data=self.sub_nHI_grid[i])
            grp_gas_grid.create_dataset(str(i),data=self.sub_gas_grid[i])
        f.close()

    def __del__(self):
        """Delete big arrays"""
        try:
            del self.sub_gas_grid
        except AttributeError:
            pass
        try:
            del self.sub_nHI_grid
        except AttributeError:
            pass
        del self.sub_mass
#         del self.sub_gas_mass
        del self.sub_cofm
        del self.sub_radii
        del self.ngrid
        del self.ind

    def set_nHI_grid(self):
        """Set up the grid around each halo where the HI is calculated.
        """
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
            #Perform the grid interpolation
            [self.sub_gridize_single_file(ii,ipos,smooth,irho,self.sub_gas_grid,irhoH0,self.sub_nHI_grid) for ii in xrange(0,self.nhalo)]
            #Explicitly delete some things.
            del ipos
            del irhoH0
            del irho
            del smooth
        [np.log1p(grid,grid) for grid in self.sub_gas_grid]
        [np.log1p(grid,grid) for grid in self.sub_nHI_grid]
        #No /= in list comprehensions...  :|
        for i in xrange(0,self.nhalo):
            self.sub_gas_grid[i]/=np.log(10)
            self.sub_nHI_grid[i]/=np.log(10)
        return

    def sub_gridize_single_file(self,ii,ipos,ismooth,irho,sub_gas_grid,irhoH0,sub_nHI_grid):
        """Helper function for sub_gas_grid and sub_nHI_grid
            that puts data arrays loaded from a particular file onto the grid.
            Arguments:
                pos - Position array
                rho - Density array to be interpolated
                smooth - Smoothing lengths
                sub_grid - Grid to add the interpolated data to
        """
        #Linear dimension of each cell in cm:
        #               kpc/h                   1 cm/kpc
        epsilon=2.*self.sub_radii[ii]/(self.ngrid[ii])*self.UnitLength_in_cm/self.hubble
        #Find particles near each halo
        sub_pos=self.sub_cofm[ii]
        indx=np.where(np.abs(ipos[:,0]-sub_pos[0]) < self.sub_radii[ii])
        pposx=ipos[indx]
        indz=np.where(np.all(np.abs(pposx[:,1:3]-sub_pos[1:3]) < self.sub_radii[ii],axis=1))
        if np.size(indz) == 0:
            return
        #coords in grid units
        coords=fieldize.convert_centered(pposx[indz]-sub_pos,self.ngrid[ii],2*self.sub_radii[ii])
        #NH0
        smooth = (ismooth[indx])[indz]
        #Convert smoothing lengths to grid coordinates.
        smooth*=(self.ngrid[ii]/(2*self.sub_radii[ii]))
        if self.once:
            print "Av. smoothing length is ",np.mean(smooth)*2*self.sub_radii[ii]/self.ngrid[ii]," kpc/h ",np.mean(smooth), "grid cells"
            self.once=False
        rho=((irho[indx])[indz])*(epsilon/(1+self.redshift)**2)
        fieldize.cic_str(coords,rho,sub_gas_grid[ii],smooth)
        rhoH0=(irhoH0[indx])[indz]*(epsilon/(1+self.redshift)**2)
        fieldize.cic_str(coords,rhoH0,sub_nHI_grid[ii],smooth)
        return

    def find_wanted_halos(self):
        """When handed a halo catalogue, remove from it the halos that are close to other, larger halos.
        Select halos via their M_200 mass, defined in terms of the critical density."""
        #Array to note the halos we don't want
        #Get halo catalog
        subs=readsubf.subfind_catalog(self.snap_dir,self.snapnum,masstab=True,long_ids=True)
        #This is rho_c in units of h^-1 1e10 M_sun (kpc/h)^-3
        rhom = 2.78e+11* self.omegam / 1e10 / (1e3**3)
        #Mass of an SPH particle, in units of 1e10 M_sun, x omega_m/ omega_b.
        target_mass = self.box**3 * rhom / self.npart[0]
        #Get list of halos resolved, using a mass cut; cuts off at about 2e9 for 512**3 particles.
        ind=np.where(subs.group_m_crit200 > self.minpart*target_mass)
        #Store the indices of the halos we are using
        #Get particle center of mass, use group catalogue.
        sub_cofm=np.array(subs.group_pos[ind])
        #halo masses in M_sun/h: use M_200
        sub_mass=np.array(subs.group_m_crit200[ind])*self.UnitMass_in_g/self.SolarMass_in_g
        #r200 in kpc.
        sub_radii = np.array(subs.group_r_crit200[ind])
        #Gas mass in M_sun/h
#         sub_gas_mass=np.array(subs.sub_masstab[ind][:,0])*self.UnitMass_in_g/self.SolarMass_in_g
        del subs
        #For each halo
        ind2=np.where([is_masked(ii,sub_mass,sub_cofm,sub_radii) for ii in xrange(0,np.size(sub_mass))])
        ind=(np.ravel(ind)[ind2],)
        sub_mass=sub_mass[ind2]
        sub_cofm=sub_cofm[ind2]
        sub_radii=sub_radii[ind2]
#         sub_gas_mass=sub_gas_mass[ind2]
        return (ind, sub_mass,sub_cofm,sub_radii)

    def get_sigma_DLA_halo(self,halo,DLA_cut):
        """Get the DLA cross-section for a single halo.
        This is defined as the area of all the cells with column density above 10^DLA_cut (10^20.3) cm^-2.
        Returns result in (kpc/h)^2."""
        cell_area=(2.*self.sub_radii[halo]/self.ngrid[halo])**2
        sigma_DLA = np.shape(np.where(self.sub_nHI_grid[halo] > DLA_cut))[1]*cell_area
        return sigma_DLA

    def get_sigma_DLA(self,DLA_cut=20.3):
        """Get the DLA cross-section from the neutral hydrogen column densities found in this class.
        This is defined as the area of all the cells with column density above 10^DLA_cut (10^20.3) cm^-2.
        Returns result in (kpc/h)^2."""
        sigma_DLA = np.array([ self.get_sigma_DLA_halo(halo,DLA_cut) for halo in xrange(0,np.size(self.ngrid))])
        return sigma_DLA


    def sigma_DLA_fit(self,M,DLA_cut=20.3,halo_mass=None):
        """Returns sigma_DLA(M) for the linear regression fit"""
        #Fit to the DLA abundance
        if halo_mass == None:
            halo_mass = self.sub_mass
        (self.pow_break,self.beta,self.gamma,self.alpha)=self.do_power_fit(halo_mass,DLA_cut)
        return self.eval_fit(M,self.alpha,self.beta,self.gamma,self.pow_break)

    def eval_fit(self,M,alpha,beta,gamma,pow_break):
        """Evaluate the fit generated by do_power_fit, below"""
        logmass=np.log10(M)-pow_break
        ind=np.where(logmass < 0)
        fit=10**(alpha*logmass+beta)
        fit[ind]=10**(gamma*logmass+beta)
        return fit

    import brokenpowerfit as br

    def do_power_fit(self,masses,DLA_cut=20.3):
        """Finds the parameters of a power law fit. Helper for sigma_DLA_fit"""
        #Fit to the DLA abundance
        s_DLA=self.get_sigma_DLA(DLA_cut)
        ind=np.where((s_DLA > 0.) * (masses > 0))
        logmass=np.log10(masses[ind])
        logsigma=np.log10(s_DLA[ind])
        if np.size(logsigma) == 0:
            return (0,0,0)
        return self.br.brokenpowerfit(logmass, logsigma)

    def identify_eq_halo(self,mass,pos,maxmass=0.10,maxpos=20.):
        """Given a mass and position, identify the
        nearest halo. Maximum tolerances are in maxmass and maxpos.
        maxmass is a percentage difference
        maxpos is an absolute difference.
        Returns an array index for self.sub_mass"""
        #First find nearby masses
        dmass=np.abs(self.sub_mass-mass)
        ind = np.where(dmass < mass*maxmass)
        #Find which of these are also nearby in positions
        ind2=np.where(np.all(np.abs(self.sub_cofm[ind]-pos) < maxpos,axis=1))
        #Is the intersection of these two sets non-zero?
        #Return the nearest mass halo
        if np.size(ind2):
            ind3=np.where(np.min(dmass[ind][ind2]) == dmass[ind][ind2])
            return ind[0][ind2][ind3]
        else:
            return np.array([])

    def get_stacked_radial_profile(self,minM,maxM,minR,maxR,gas_grid=False):
        """Stacks several radial profiles in mass bins"""
        ind = np.where(np.logical_and(self.sub_mass > minM, self.sub_mass < maxM))
        stack_element=[self.get_radial_profile(ii, minR, maxR,gas_grid) for ii in np.ravel(ind)]
        return np.mean(stack_element)

    def get_radial_profile(self,halo,minR,maxR,gas_grid=False):
        """Returns the nHI density summed radially
           (but really in Cartesian coordinates).
           So returns R_HI (cm^-1).
           Should use bins in r significantly larger
           than the grid size.
        """
        #This is an integral over an annulus in Cartesians
        if gas_grid:
            grid=self.sub_gas_grid[halo]
        else:
            grid=self.sub_nHI_grid[halo]

        #Find r in grid units:
        total=0.
        gminR=minR/(2.*self.sub_radii[halo])*self.ngrid[halo]
        gmaxR=maxR/(2.*self.sub_radii[halo])*self.ngrid[halo]
        cen=self.ngrid[halo]/2.
        #Broken part of the annulus:
        for x in xrange(-int(gminR),int(gminR)):
            miny=int(np.sqrt(gminR**2-x**2))
            maxy=int(np.sqrt(gmaxR**2-x**2))
            total+=np.sum(10**grid[x+self.ngrid[halo]/2,(cen+miny):(cen+maxy)])
            total+=np.sum(10**grid[x+self.ngrid[halo]/2,(cen-maxy):(cen-miny)])
        #Complete part of annulus
        for x in xrange(int(gminR),int(gmaxR)):
            maxy=int(np.sqrt(gmaxR**2-x**2)+cen)
            miny=int(-np.sqrt(gmaxR**2-x**2)+cen)
            total+=np.sum(10**grid[x+cen,miny:maxy])
            total+=np.sum(10**grid[-x+cen,miny:maxy])
        return total*((2.*self.sub_radii[halo])/self.ngrid[halo]*self.UnitLength_in_cm)

    def get_halo_fit_parameters(self):
        """Get an array of M_0 r_0 for a list of masses"""
        minM = np.min(self.sub_mass)
        maxM = np.max(self.sub_mass)
        Mbins = np.logspace(np.log10(minM), np.log10(maxM),11)
        Mbinc = [(Mbins[i+1]+Mbins[i])/2 for i in xrange(0,np.size(Mbins)-1)]
        M0 = np.zeros(np.shape(Mbinc))
        r0 = np.zeros(np.shape(Mbinc))
        for i in xrange(0,np.size(Mbins)-1):
            (r0[i], M0[i])=self.get_halo_scale_length(Mbins[i],Mbins[i+1])
        return (Mbinc, M0, r0)

    import mpfit
    def get_halo_scale_length(self,minM,maxM):
        """Get the fitted scale length of a stack of halos.
        Fits a halo profile of M_0/(1+r/r_0)^2 to the central part with r_0 free."""
        #Use sufficiently large bins
        minR=0.
        maxR=30.
        space=2.*self.sub_radii[0]/self.ngrid[0]
        if maxR/20. > space:
            Rbins=np.linspace(minR,maxR,20)
        else:
            Rbins=np.concatenate((np.array([minR,]),np.linspace(minR+np.ceil(1.5*space),maxR+space,maxR/np.ceil(space))))
        Rprof=[self.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1]) for i in xrange(0,np.size(Rbins)-1)]
        Rbinc = [(Rbins[i+1]+Rbins[i])/2 for i in xrange(0,np.size(Rbins)-1)]
        #Central density
        M0 = Rprof[0]
        if M0 < 1:
            raise Exception
        err = np.ones(np.shape(Rprof))
        pinit = [10,M0]
        #Non-changing parameters to mpfitfun
        params={'xax':Rbinc,'data':Rprof,'err':err,}
        #Do fit
        mp = self.mpfit.mpfit(self.mpfitfun,xall=pinit,functkw=params,quiet=True)
        #Return M0, R0
        return mp.params

    def mpfitfun(self,p,fjac=None,xax=None,data=None,err=None):
        """This function returns a status flag (0 for success)
        and the weighted deviations between the model and the data
            Parameters:
            p[0] - r0
            p[1] - M0"""
        fit=p[1]/(1+xax/p[0])**2
        return [0,np.ravel((fit-data)/err)]


    def get_halo_central_density(self,halo):
        """Returns the HI column density at the center of the halo"""
        grid = self.sub_nHI_grid[halo]
        dims = np.shape(grid)
        center=tuple(np.array(dims)/2)
        #3x3 square at center of grid
        return np.mean(grid[center[0]-1:center[0]+2,center[1]-1:center[1]+2])

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
        and n_DLA(N) = number of absorbers per sightline in this column density bin.
                     1 sightline is defined to be one grid cell.
                     So this is (cells in this bins) / (no. of cells)
        ie, f(N) = n_DLA / ΔN / ΔX
        Note f(N) has dimensions of cm^2, because N has units of cm^-2 and X is dimensionless.

        Parameters:
            dlogN - bin spacing to aim for (may not actually be reached)
            minN - minimum log N
            maxN - maximum log N

        Returns:
            (NHI, f_N_table) - N_HI (binned in log) and corresponding f(N)
        """
        NHI_table = np.log10(np.logspace(minN, maxN,(maxN-minN)/dlogN,endpoint=True))
        #Grid size (in cm^2)
        dX=self.absorption_distance()
        tot_cells = np.sum(self.ngrid**2)
        array=np.array([np.histogram(np.ravel(grid),NHI_table) for grid in self.sub_nHI_grid])
        tot_f_N = np.sum(array[:,0])
        NHI_table = (10.)**array[0,1]
        #To compensate for any rounding
        dlogN_real = (np.log10(NHI_table[-1])-np.log10(NHI_table[0]))/(np.size(NHI_table)-1)
        tot_f_N=(tot_f_N)/((NHI_table[0:-1]*(10.)**dlogN_real-NHI_table[0:-1])*dX*tot_cells)
        return (NHI_table[0:-1]*(10.)**(dlogN_real/2.), tot_f_N)

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
    def __init__(self,snap_dir,snapnum,reload_file=False,skip_grid=None,savefile=None):
        if savefile==None:
            savefile_s=path.join(snap_dir,"snapdir_"+str(snapnum).rjust(3,'0'),"boxhi_grid.hdf5")
        else:
            savefile_s = savefile
        HaloHI.__init__(self,snap_dir,snapnum,minpart=-1,reload_file=reload_file,skip_grid=skip_grid,savefile=savefile_s)
        return

    def sub_gridize_single_file(self,ii,ipos,smooth,irho,sub_gas_grid,irhoH0,sub_nHI_grid):
        """Helper function for sub_gas_grid and sub_nHI_grid
            that puts data arrays loaded from a particular file onto the grid.
            Arguments:
                pos - Position array
                rho - Density array to be interpolated
                smooth - Smoothing lengths
                sub_grid - Grid to add the interpolated data to
        """
        #Linear dimension of each cell in cm:
        #               kpc/h                   1 cm/kpc
        epsilon=2.*self.sub_radii/(self.ngrid)*self.UnitLength_in_cm/self.hubble
        #coords in grid units
        coords=fieldize.convert(ipos,self.ngrid,2*self.sub_radii)
        #NH0
        if self.once:
            print "Av. smoothing length is ",np.mean(smooth)*2*self.sub_radii/self.ngrid," kpc/h ",np.mean(smooth), "grid cells"
            self.once=False
        rho=(irho)*(epsilon/(1+self.redshift)**2)
        fieldize.cic_str(coords,rho,sub_gas_grid[ii],smooth)
        rhoH0=(irhoH0)*(epsilon/(1+self.redshift)**2)
        fieldize.cic_str(coords,rhoH0,sub_nHI_grid[ii],smooth)
        return

