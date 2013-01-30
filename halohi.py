# vim: set fileencoding=utf-8
"""Module for creating the DLA hydrogen density plots, as found in Tescari & Viel,
and Nagamine, Springel and Hernquist, 2003.

Classes:
    TotalHaloHI - Finds the average HI fraction in a halo
    HaloHI - Creates a grid around the halo center with the HI fraction calculated at each grid cell
"""
#testing 123
import numpy as np
import numexpr as ne
import readsubf
import readsubfHDF5
import hdfsim
import h5py
import math
import os.path as path
import cold_gas
import halo_mass_function
import fieldize
import hsml
import scipy.integrate as integ
import scipy.stats
import mpfit

def is_masked(halo,sub_mass,sub_cofm, sub_radii):
    """Find out whether a halo is a mere satellite and if so mask it"""
    near=np.where(np.all((np.abs(sub_cofm[:,:]-sub_cofm[halo,:]) < sub_radii[halo]),axis=1))
    #If there is a larger halo nearby, mask this halo
    return np.size(np.where(sub_mass[near] > sub_mass[halo])) == 0

def calc_binned_median(bin_edge,xaxis,data):
    """Calculate the median value of an array in some bins"""
    media = np.zeros(np.size(bin_edge)-1)
    for i in xrange(0,np.size(bin_edge)-1):
        ind = np.where((xaxis > bin_edge[i])*(xaxis < bin_edge[i+1]))
        if np.size(ind) > 0:
            media[i] = np.median(data[ind])
    return media


def calc_binned_percentile(bin_edge,xaxis,data,per=75):
    """Calculate the percentile value of an array in some bins.
    per is the percentile at which to extract it. """
    percen = np.zeros(np.size(bin_edge)-1)
    for i in xrange(0,np.size(bin_edge)-1):
        ind = np.where((xaxis > bin_edge[i])*(xaxis < bin_edge[i+1]))
        if np.size(ind) > 0:
            percen[i] = scipy.stats.scoreatpercentile(data[ind],per)
    return percen

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
        #This is rho_c in units of h^-1 1e10 M_sun (kpc/h)^-3
        rhom = 2.78e+11* self.omegam / 1e10 / (1e3**3)
        #Mass of an SPH particle, in units of 1e10 M_sun, x omega_m/ omega_b.
        target_mass = self.box**3 * rhom / self.npart[0]
        #Get halo catalog
        try:
            subs=readsubf.subfind_catalog(self.snap_dir,self.snapnum,masstab=True,long_ids=True)
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
#             sub_gas_mass=np.array(subs.sub_masstab[ind][:,0])*self.UnitMass_in_g/self.SolarMass_in_g
            del subs
        except IOError:
        # We might have the halo catalog stored in the new format, which is HDF5.
            subs=readsubfHDF5.subfind_catalog(self.snap_dir,self.snapnum,long_ids=True)
            #Get list of halos resolved, using a mass cut; cuts off at about 2e9 for 512**3 particles.
            ind=np.where(subs.Group_M_Crit200 > self.minpart*target_mass)
            #Store the indices of the halos we are using
            #Get particle center of mass, use group catalogue.
            sub_cofm=np.array(subs.GroupPos[ind])
            #halo masses in M_sun/h: use M_200
            sub_mass=np.array(subs.Group_M_Crit200[ind])*self.UnitMass_in_g/self.SolarMass_in_g
            #r200 in kpc.
            sub_radii = np.array(subs.Group_R_Crit200[ind])
            #Gas mass in M_sun/h
#             sub_gas_mass=np.array(subs.sub_masstab[ind][:,0])*self.UnitMass_in_g/self.SolarMass_in_g
            del subs
        # We might have the halo catalog stored in the new format, which is HDF5.

        # blehh
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
    def __init__(self,snap_dir,snapnum,minpart=400,grid_width=-1.,reload_file=False,skip_grid=None,savefile=None):
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
                print "Will try to reload"
                #raise KeyError("reloading")
            #First try to load from a file
            f=h5py.File(self.savefile,'r')
            grid_file=f["HaloData"]
            if not (grid_file.attrs["minpart"] == self.minpart):
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

            if minpart == -1:
                #global grid
                self.nhalo = 1
                self.sub_radii=np.array([self.box/2.])
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
                self.sub_radii=np.array([self.box/2.])
            else:
                print "Found ",self.nhalo," halos with > ",minpart,"particles"
                print "Smallest has mass: ", np.min(self.sub_mass)
                print "Largest has mass: ", np.max(self.sub_mass)
            #Set ngrid to be the gravitational softening length
            if grid_width == -1.:
                self.grid_radii = self.sub_radii
            elif grid_width > 0:
                self.grid_radii = grid_width * np.ones(self.nhalo)
            self.ngrid=np.array([int(np.ceil(40*self.npart[1]**(1./3)/self.box*2*rr)) for rr in self.grid_radii])
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

    def get_H2_frac(self,nHI):
        """Get the molecular fraction for neutral gas"""
        fH2 = 1./(1+(0.1/nHI)**(0.92*5./3.)*35**0.92)
        fH2[np.where(nHI < 0.1)] = 0
        return fH2

    #rmate test

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
            #Re-use sub_gas_grid to be metallicity
#             irho=np.array(bar["Metallicity"],dtype=np.float64)
            protonmass=1.66053886e-24
            hy_mass = 0.76 # Hydrogen massfrac
            # gas density in hydrogen atoms/cm^3
            irho*=(hy_mass/protonmass)
            f.close()
            #Re-use sub_gas_grid to be molecular hydrogen
#             fH2 = self.get_H2_frac(irhoH0)
#             irhoH2 = irhoH0 * fH2
#             irhoH0 *= (1-fH2)
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

    def sub_gridize_single_file(self,ii,ipos,ismooth,irho,sub_gas_grid,irhoH0,sub_nHI_grid,weights=None):
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
        epsilon=2.*self.grid_radii[ii]/(self.ngrid[ii])*self.UnitLength_in_cm/self.hubble
        #Find particles near each halo
        sub_pos=self.sub_cofm[ii]
        xpos = sub_pos[0]
        xxpos = ipos[:,0]
        grid_radius = self.grid_radii[ii]
#   indz=np.where(ne.evaluate("prod(abs(ipos-sub_pos) < self.sub_radii[ii],axis=1)"))
#         indx=np.where(np.abs(ipos[:,0]-sub_pos[0]) < self.sub_radii[ii])
        indx=np.where(ne.evaluate("abs(xxpos-xpos) < grid_radius"))
        pposx=ipos[indx]
        indz=np.where(np.all(np.abs(pposx[:,1:3]-sub_pos[1:3]) < self.grid_radii[ii],axis=1))
        if np.size(indz) == 0:
            return
        #coords in grid units
        coords=fieldize.convert_centered(pposx[indz]-sub_pos,self.ngrid[ii],2*self.grid_radii[ii])
        #NH0
        smooth = (ismooth[indx])[indz]
        #Convert smoothing lengths to grid coordinates.
        smooth*=(self.ngrid[ii]/(2*self.grid_radii[ii]))
        if self.once:
            print ii," Av. smoothing length is ",np.mean(smooth)*2*self.grid_radii[ii]/self.ngrid[ii]," kpc/h ",np.mean(smooth), "grid cells"
            self.once=False
        rho=((irho[indx])[indz])*epsilon*(1+self.redshift)**2
        fieldize.sph_str(coords,rho,sub_gas_grid[ii],smooth,weights=weights)
        rhoH0=(irhoH0[indx])[indz]*epsilon*(1+self.redshift)**2
        fieldize.sph_str(coords,rhoH0,sub_nHI_grid[ii],smooth,weights=weights)
        return

    def find_wanted_halos(self):
        """When handed a halo catalogue, remove from it the halos that are close to other, larger halos.
        Select halos via their M_200 mass, defined in terms of the critical density."""
        #This is rho_c in units of h^-1 1e10 M_sun (kpc/h)^-3
        rhom = 2.78e+11* self.omegam / 1e10 / (1e3**3)
        #Mass of an SPH particle, in units of 1e10 M_sun, x omega_m/ omega_b.
        target_mass = self.box**3 * rhom / self.npart[0]
        #Get halo catalog
        try:
            subs=readsubf.subfind_catalog(self.snap_dir,self.snapnum,masstab=True,long_ids=True)
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
#             sub_gas_mass=np.array(subs.sub_masstab[ind][:,0])*self.UnitMass_in_g/self.SolarMass_in_g
            del subs
        except IOError:
        # We might have the halo catalog stored in the new format, which is HDF5.
            subs=readsubfHDF5.subfind_catalog(self.snap_dir,self.snapnum,long_ids=True)
            #Get list of halos resolved, using a mass cut; cuts off at about 2e9 for 512**3 particles.
            ind=np.where(subs.Group_M_Crit200 > self.minpart*target_mass)
            #Store the indices of the halos we are using
            #Get particle center of mass, use group catalogue.
            sub_cofm=np.array(subs.GroupPos[ind])
            #halo masses in M_sun/h: use M_200
            sub_mass=np.array(subs.Group_M_Crit200[ind])*self.UnitMass_in_g/self.SolarMass_in_g
            #r200 in kpc.
            sub_radii = np.array(subs.Group_R_Crit200[ind])
            #Gas mass in M_sun/h
#             sub_gas_mass=np.array(subs.sub_masstab[ind][:,0])*self.UnitMass_in_g/self.SolarMass_in_g
            del subs
        # We might have the halo catalog stored in the new format, which is HDF5.

        #For each halo
        ind2=np.where([is_masked(ii,sub_mass,sub_cofm,sub_radii) for ii in xrange(0,np.size(sub_mass))])
        ind=(np.ravel(ind)[ind2],)
        sub_mass=sub_mass[ind2]
        sub_cofm=sub_cofm[ind2]
        sub_radii=sub_radii[ind2]
#         sub_gas_mass=sub_gas_mass[ind2]
        return (ind, sub_mass,sub_cofm,sub_radii)

    def get_sigma_DLA_halo(self,halo,DLA_cut,DLA_upper_cut=42.):
        """Get the DLA cross-section for a single halo.
        This is defined as the area of all the cells with column density above 10^DLA_cut (10^20.3) cm^-2.
        Returns result in (kpc)^2."""
        #Linear dimension of cell in kpc.
        epsilon=2.*self.grid_radii[halo]/(self.ngrid[halo])/self.hubble
        cell_area=epsilon**2 #(2.*self.grid_radii[halo]/self.ngrid[halo])**2
        sigma_DLA = np.shape(np.where((self.sub_nHI_grid[halo] > DLA_cut)*(self.sub_nHI_grid[halo] < DLA_upper_cut)))[1]*cell_area
        return sigma_DLA

    def get_sigma_DLA(self,DLA_cut=20.3,DLA_upper_cut=42.):
        """Get the DLA cross-section from the neutral hydrogen column densities found in this class.
        This is defined as the area of all the cells with column density above 10^DLA_cut (10^20.3) cm^-2.
        Returns result in (kpc)^2. Omits cells above DLA_upper_cut"""
        sigma_DLA = np.array([ self.get_sigma_DLA_halo(halo,DLA_cut,DLA_upper_cut) for halo in xrange(0,np.size(self.ngrid))])
        return sigma_DLA

    def get_sigma_DLA_binned(self,mass,DLA_cut=20.3,DLA_upper_cut=42.,sigma=95):
        """Get the median and scatter of sigma_DLA against mass."""
        sigDLA=self.get_sigma_DLA(DLA_cut,DLA_upper_cut)
        aind = np.where(sigDLA > 0)
        amed=calc_binned_median(mass, self.sub_mass[aind], sigDLA[aind])
        aupq=calc_binned_percentile(mass, self.sub_mass[aind], sigDLA[aind],sigma)-amed
        #Addition to avoid zeros
        aloq=amed - calc_binned_percentile(mass, self.sub_mass[aind], sigDLA[aind],100-sigma)
        return (amed, aloq, aupq)

    def get_mean_halo_mass(self,DLA_cut=20.3,DLA_upper_cut=42.):
        """Get the mean halo mass for DLAs"""
        gsigDLA=self.get_sigma_DLA(DLA_cut,DLA_upper_cut)
        #Generate mean halo mass
        g_mean_halo_mass = np.sum(self.sub_mass*gsigDLA)/np.sum(gsigDLA)
        return g_mean_halo_mass

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
        gminR=minR/(2.*self.grid_radii[halo])*self.ngrid[halo]
        gmaxR=maxR/(2.*self.grid_radii[halo])*self.ngrid[halo]
        cen=self.ngrid[halo]/2.
        #Broken part of the annulus:
        for x in xrange(-int(gminR),int(gminR)):
            miny=int(np.sqrt(gminR**2-x**2))
            maxy=int(np.sqrt(gmaxR**2-x**2))
            try:
                total+=np.sum(10**grid[x+self.ngrid[halo]/2,(cen+miny):(cen+maxy)])
                total+=np.sum(10**grid[x+self.ngrid[halo]/2,(cen-maxy):(cen-miny)])
            except IndexError:
                pass
        #Complete part of annulus
        for x in xrange(int(gminR),int(gmaxR)):
            maxy=int(np.sqrt(gmaxR**2-x**2)+cen)
            miny=int(-np.sqrt(gmaxR**2-x**2)+cen)
            try:
                total+=np.sum(10**grid[x+cen,miny:maxy])
                total+=np.sum(10**grid[-x+cen,miny:maxy])
            except IndexError:
                pass
        return total*((2.*self.grid_radii[halo])/self.ngrid[halo]*self.UnitLength_in_cm)



    def get_sDLA_fit(self):
        """Fit an Einasto profile based function to sigma_DLA as binned."""
        minM = np.min(self.sub_mass)
        maxM = np.max(self.sub_mass)
        bins=30
        mass=np.logspace(np.log10(minM),np.log10(maxM),num=bins)
        bin_mass = np.array([(mass[i]+mass[i+1])/2. for i in xrange(0,np.size(mass)-1)])
        (sDLA,loq,upq)=self.get_sigma_DLA_binned(mass,sigma=68)
        (sLLS,loqLL,upqLL)=self.get_sigma_DLA_binned(mass,DLA_cut=17.,sigma=68)
        indLL = np.where((sLLS > 0)*(loqLL+upqLL > 0))
        errLL = (upqLL[indLL]+loqLL[indLL])/2.
        ind = np.where((sDLA > 0)*(loq+upq > 0))
        err = (upq[ind]+loq[ind])/2.
        #Arbitrary large values if err is zero
        pinit = [0.5,32.,30,0,2]
        #Non-changing parameters to mpfitfun
#         params={'xax':bin_mass[ind],'data':np.log10(sDLA[ind]),'err':np.log10(err)}
        params={'xax':bin_mass[ind],'data':np.log10(sDLA[ind]),'err':np.log10(err),'errLL':np.log10(errLL),'dataLL':np.log10(sLLS[indLL]),'xaxLL':bin_mass[indLL]}
        #Do fit
        mp = mpfit.mpfit(self.mpfitfun,xall=pinit,functkw=params,quiet=True)
        #Return M0, R0
        return mp.params

    def mpfitfun(self,p,fjac=None,xax=None,data=None,err=None,errLL=None,dataLL=None,xaxLL=None):
#     def mpfitfun(self,p,fjac=None,xax=None,data=None,err=None):
        """This function returns a status flag (0 for success)
        and the weighted deviations between the model and the data
            Parameters:
            p[0] - rho_0 a
            p[1] - rho_0 b
            p[2] - r0 a
            p[3] - r0 b
        """
        fit=np.log10(self.sDLA_analytic(xax,p))
        fit2=np.log10(self.sDLA_analytic(xaxLL,p,DLA_cut=17.))
        return [0,np.concatenate([np.ravel((fit-data)/err),np.ravel((fit2-dataLL)/errLL)])]

    def absorption_distance(self):
        """Compute X(z), the absorption distance per sightline (eq. 9 of Nagamine et al 2003)
        in dimensionless units."""
        #h * 100 km/s/Mpc in h/s
        h100=3.2407789e-18
        # in cm/s
        light=2.9979e10
        #Units: h/s   s/cm                        kpc/h      cm/kpc
        return h100/light*(1+self.redshift)**2*self.box*self.UnitLength_in_cm

    def column_density_function(self,dlogN=0.2, minN=20.3, maxN=30., maxM=13,minM=9,grids=None):
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
        if grids == None:
            grids = self.sub_nHI_grid
        elif grids == 1:
            grids = self.sub_gas_grid
        NHI_table = np.arange(minN, maxN, dlogN)
        bin_center = np.array([(NHI_table[i]+NHI_table[i+1])/2. for i in range(0,np.size(NHI_table)-1)])
        deltaNHI =  np.array([10**NHI_table[i+1]-10**NHI_table[i] for i in range(0,np.size(NHI_table)-1)])
        #Grid size (in cm^2)
        dX=self.absorption_distance()
        ind = np.where((self.sub_mass < 10.**maxM)*(self.sub_mass > 10.**minM))
        tot_cells = np.sum(self.ngrid**2)
        array=np.array([np.histogram(np.ravel(grid),NHI_table) for grid in grids[ind]])
        tot_f_N = np.sum(array[:,0])
        tot_f_N=(tot_f_N)/(deltaNHI*dX*tot_cells)
        return (10**bin_center, tot_f_N)

    def omega_DLA(self, maxN):
        """Compute Omega_DLA, defined as:
            Ω_DLA = m_p H_0/(c f_HI rho_c) \int_10^20.3^Nmax  f(N,X) N dN
        """
        (NHI_table, f_N) = self.column_density_function(minN=20.3,maxN=maxN)
        dNHI_table = np.arange(20.3, maxN, 0.2)
        deltaNHI =  np.array([10**dNHI_table[i+1]-10**dNHI_table[i] for i in range(0,np.size(dNHI_table)-1)])
        omega_DLA=np.sum(NHI_table*f_N*deltaNHI)
        h100=3.2407789e-18*self.hubble
        light=2.9979e10
        rho_crit=3*h100**2/(8*math.pi*6.672e-8)
        protonmass=1.66053886e-24
        hy_mass = 0.76 # Hydrogen massfrac
        omega_DLA*=(h100/light)*(protonmass/hy_mass)/rho_crit
        return 1000*omega_DLA

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

    def sDLA_analytic(self,M,params, DLA_cut=20.3):
        """An analytic fit to the DLA radius,
        based on a power law."""
        a = params[0]
        b = params[1]
        ra = params[2]
        e = params[4]
        br = 10.5
        n=5.
        d = params[3]/10**(DLA_cut/n)
        N0 = 10.**(a*(np.log10(M)-br))
        sDLA = (d*N0**e+N0)*10**((b-DLA_cut)/n) -ra
        ind = np.where(sDLA <= 0)
        if np.size(ind) > 0:
            try:
                sDLA[ind]=1e-50
            except TypeError:
                #This is necessary in case RDLA is a single float, not an array
                sDLA=1e-50
        return sDLA

    def drdz(self,zz):
        """Calculates dr/dz in a flat cosmology in units of cm/h"""
        #Speed of light in cm/s
        light=2.9979e10
        #h * 100 km/s/Mpc in h/s
        h100=3.2407789e-18
        #       cm/s   s/h   =>
        return light/h100*np.sqrt(self.omegam*(1+zz)**3+self.omegal)

    def get_mean_halo_mass(self,params,mass=5e8):
        """Get the mean halo mass for DLAs"""
#         gsigDLA=self.get_sigma_DLA(DLA_cut,DLA_upper_cut)
        #Generate mean halo mass
#         g_mean_halo_mass = np.sum(self.sub_mass*gsigDLA)/np.sum(gsigDLA)
        maxmass=np.log10(np.max(self.sub_mass))
        abund=self.get_N_DLA_dz(params,mass,maxmass)/(self.drdz(self.redshift)/self.UnitLength_in_cm)
        abund_mass = integ.quad(self.mass_integrand,np.log10(mass),maxmass, epsrel=1e-2,args=(params,))
        return abund_mass[0]/abund

    def mass_integrand(self,log10M,params):
        """Integrand for above"""
        M=10**log10M
        return M*self.NDLA_integrand(log10M,params)

    def get_N_DLA_dz(self,params, mass=1e9,maxmass=12.5):
        """Get the DLA number density as a function of redshift, defined as:
        d N_DLA / dz ( > M, z) = dr/dz int^\infinity_M n_h(M', z) sigma_DLA(M',z) dM'
        where n_h is the Sheth-Torman mass function, and
        sigma_DLA is a power-law fit to self.sigma_DLA.
        Parameters:
            lower_mass in M_sun/h.
        """
        try:
            self.halo_mass.dndm(mass)
        except AttributeError:
            #Halo mass function object
            self.halo_mass=halo_mass_function.HaloMassFunction(self.redshift,omega_m=self.omegam, omega_l=self.omegal, hubble=self.hubble,log_mass_lim=(7,15))
        result = integ.quad(self.NDLA_integrand,np.log10(mass),maxmass, epsrel=1e-2,args=(params,))
        #drdz is in cm/h, while the rest is in kpc/h, so convert.
        return self.drdz(self.redshift)*result[0]/self.UnitLength_in_cm

    def NDLA_integrand(self,log10M,params):
        """Integrand for above"""
        M=10**log10M
        #sigma_DLA_analytic is in kpc^2, while halo_mass is in h^4 M_sun^-1 Mpc^(-3), and M is in M_sun/h.
        #Output therefore in kpc/h
        return self.sDLA_analytic(M,params,20.3)*self.hubble**2*M/(10**9)*self.halo_mass.dndm(M)

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
        fieldize.sph_str(coords,rho,sub_gas_grid[ii],smooth)
        rhoH0=(irhoH0)*(epsilon/(1+self.redshift)**2)
        fieldize.sph_str(coords,rhoH0,sub_nHI_grid[ii],smooth)
        return

    def column_density_function(self,dlogN=0.2, minN=20.3, maxN=30., maxM=13,minM=9,grids=None):
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
        if grids == None:
            grids = self.sub_nHI_grid
        elif grids == 1:
            grids = self.sub_gas_grid
        NHI_table = np.arange(minN, maxN, dlogN)
        bin_center = np.array([(NHI_table[i]+NHI_table[i+1])/2. for i in range(0,np.size(NHI_table)-1)])
        deltaNHI =  np.array([10**NHI_table[i+1]-10**NHI_table[i] for i in range(0,np.size(NHI_table)-1)])
        #Grid size (in cm^2)
        dX=self.absorption_distance()
        #This is for whole box grids
        tot_f_N=np.histogram(grids,NHI_table)
        tot_cells = np.size(grids)
        tot_f_N=(tot_f_N[0])/(deltaNHI*dX*tot_cells)
        return (10**bin_center, tot_f_N)


class VelocityHI(HaloHI):
    """Class for computing velocity diagrams"""
    def __init__(self,snap_dir,snapnum,minpart,reload_file=False,skip_grid=None,savefile=None):
        if savefile==None:
            savefile_s=path.join(snap_dir,"snapdir_"+str(snapnum).rjust(3,'0'),"velocity_grid.hdf5")
        else:
            savefile_s = savefile
        HaloHI.__init__(self,snap_dir,snapnum,minpart=minpart,reload_file=reload_file,skip_grid=skip_grid,savefile=savefile_s)
        return

    def set_nHI_grid(self):
        """Set up the grid around each halo where the velocity HI is calculated.
        """
        star=cold_gas.StarFormation(hubble=self.hubble)
        self.once=True
        #This is the real HI grid
        nHI_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
        #Now grid the HI for each halo
        for fnum in xrange(0,500):
            try:
                f=hdfsim.get_file(self.snapnum,self.snap_dir,fnum)
            except IOError:
                break
            print "Starting file ",fnum
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"],dtype=np.float64)
            smooth = hsml.get_smooth_length(bar)
            # Velocity in cm/s
            vel = np.array(bar["Velocities"],dtype=np.float64)*self.UnitVelocity_in_cm_per_s
            #We will weight by neutral mass per cell
            irhoH0 = star.get_reproc_rhoHI(bar)
            irho=np.array(bar["Density"],dtype=np.float64)*(self.UnitMass_in_g/self.UnitLength_in_cm**3)*self.hubble**2
            #HI * Cell Mass, internal units
            mass = np.array(bar["Masses"],dtype=np.float64)*irhoH0/irho
            f.close()
            #Perform the grid interpolation
            #sub_gas_grid is x velocity
            #sub_nHI_grid is y velocity
            [self.sub_gridize_single_file(ii,ipos,smooth,vel[:,1]*mass,self.sub_gas_grid,vel[:,2]*mass,self.sub_nHI_grid,mass) for ii in xrange(0,self.nhalo)]
            #Find the HI density also, so that we can discard
            #velocities in cells that are not DLAs.
            [self.sub_gridize_single_file(ii,ipos,smooth,irhoH0,nHI_grid,np.zeros(np.size(irhoH0)),nHI_grid) for ii in xrange(0,self.nhalo)]
            #Explicitly delete some things.
            del ipos
            del irhoH0
            del irho
            del smooth
            del mass
            del vel
        #No /= in list comprehensions...  :|
        #Average over z
        for i in xrange(0,self.nhalo):
            self.sub_gas_grid[i]/=self.sub_radii[i]
            self.sub_nHI_grid[i]/=self.sub_radii[i]
            ind = np.where(nHI_grid[i] < 10**20.3)
            self.sub_nHI_grid[i][ind] = 0
            self.sub_gas_grid[i][ind] = 0
        return

