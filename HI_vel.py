"""
Module for computing velocity profile, similar to how they are defined in Rudie et al. 2012 (e.g. see their Figures 5,6)
"""

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

# Take all gas within 3D box around each galaxy. (side of box is 1 physical Mpc; corresponds to (1+z) comoving Mpc)
# For each gas particle in the box, compute the true neutral fraction using Simeon's methods
# Mimic line-of-sight observations by only considering the z-component of the velocity
# Histogram these z-velocities 

class HIVel:

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

        #regenerate from the raw data
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

        self.histogram_velocities()
        #Set ngrid to be the gravitational softening length
        #self.ngrid=np.array([int(np.ceil(40*self.npart[1]**(1./3)/self.box*2*rr)) for rr in self.sub_radii])
        #self.sub_nHI_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
        #self.sub_gas_grid=np.array([np.zeros([self.ngrid[i],self.ngrid[i]]) for i in xrange(0,self.nhalo)])
        #self.set_nHI_grid()


    def histogram_velocities(self):
        """Histogram velocities of neutral HI around halo
        """
        star=cold_gas.StarFormation(hubble=self.hubble)
        
        #Now get the gas data from the snapshot:
        HI_pos = np.empty([0,3])
        #HI_vel = np.empty([0,3])
        HI_LoS_vel = np.empty([0])
        HI_mass = np.empty([0])
        #gas_u = np.empty([0,1])
        gas_rho = np.empty([0])

        for fnum in xrange(0,500):
            try:
                f=hdfsim.get_file(self.snapnum,self.snap_dir,fnum)
            except IOError:
                break
            print "Starting file ",fnum
            gas_dat=f["PartType0"]
            gas_pos=np.array(gas_dat["Coordinates"],dtype=np.float64)
            gas_vel=np.array(gas_dat["Velocities"],dtype=np.float64)
            gas_mass=np.array(gas_dat["Masses"],dtype=np.float64))
            #gas_u
            #gas_rho=np.array(gas_dat["Density"],dtype=np.float64)

            #Get neutral fraction (reprocessed to account for self-shielding)
            [gas_rho,inH0] = star.reproc_gas(bar)
            #del gas_rho
            f.close()

            # Append relevant arrays:
            HI_pos = np.append(HI_pos, gas_pos,axis=0)
            HI_LoS_vel = np.append(HI_vel, gas_vel[:,2])
            HI_mass = np.append(HI_mass, gas_mass*inH0)
            #gas_rho = np.append(gas_rho,)


        n_bincents = 28
        all_hist_dat = np.zeros([self.nhalo,n_bincents])

        # Now loop over halos...
        for i in np.arange(self.nhalo):
            # get all gas particles close to this halo:
            mask = self._find_particles_in_galaxy(halo_num,HI_pos)

            # Take the z-component of the velocities to be the velocities along line of sight
            LoS_vel = HI_LoS_vel[mask]-self.sub

            # Compute histogram for each galaxy of interest:
            bins = np.array(-1500.,1500.,num=(n_bincents+1))
            bin_width = bins[1]-bins[0]
            bin_cents = bins[:-1]+bin_width
            hist, bin_edges = np.histogram(LoS_vel,bins=bins,density=True)

            all_hist_dat[i,:] = hist


        # Plot stacked histogram:
        Q3 = np.arange(n_bincents)
        median = np.arange(n_bincents)
        Q1 = np.arange(n_bincents)
        
        for bin_ind in np.arange(n_bincents):
            Q3[bin_ind] = np.percentile(all_hist_dat[:,bin_ind],75)
            median[bin_ind] = np.median(all_hist_dat[:,bin_ind])
            Q1[bin_ind] = np.percentile(all_hist_dat[:,bin_ind],25)

        plt.plot(bin_cents,median)
        plt.fill_between(r_min_array, Q3, Q1, facecolor=facecolor,alpha=0.25)
        #plt.ylabel("P(\deltav,)")
        plt.xlabel("v (km/s)") # NEED TO CORRECT LOS VEL BY VEL OF HALO



        return




    def _dist(self,x,y):   
        """ Euclidean distance between 2 arrays of positions """
        return np.sqrt(np.sum((x-y)**2,axis=1))


    def _find_particles_in_galaxy(self,halo_num,part_pos,comoving_radius=1000.):
        """ Helper function that finds all particles of given type within a set distance of the halo center """

        cent_pos = self.sub_pos[halo_num]

        radius = comoving_radius

        # Fast initial cut:
        boolx = np.abs(part_pos[:,0]-cent_pos[0]) < radius
        booly = np.abs(part_pos[:,1]-cent_pos[1]) < radius
        boolz = np.abs(part_pos[:,2]-cent_pos[2]) < radius
        indxyz = np.where( np.logical_and(np.logical_and(boolx, booly),boolz))
        part_pos = part_pos[indxyz]

        # Radial cut:
        indr = np.where(self._dist(part_pos,cent_pos) < radius)

        # Galaxy mask:
        n_part = np.size(part_pos,axis=0)
        mask = np.arange(n_part)[indxyz][indr]
        return mask



