# -*- coding: utf-8 -*-
"""Module for calculating the average HI fraction in halos.
"""
import numpy as np
import halocat
import hdfsim
import os.path as path
import cold_gas


class TotalHaloHI:
    """Find the average HI fraction in a halo"""
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
            self.redshift=f["Header"].attrs["Redshift"]
            f.close()
            #Get halo catalog
            #This is rho_c in units of h^-1 1e10 M_sun (kpc/h)^-3
            rhom = 2.78e+11* self.omegam / 1e10 / (1e3**3)
            #Mass of an SPH particle, in units of 1e10 M_sun, x omega_m/ omega_b.
            target_mass = self.box**3 * rhom / self.npart[0]
            min_mass = target_mass * self.minpart
            #Get halo catalog
            (self.ind,self.mass,self.cofm,self.sub_radii)=halocat.find_wanted_halos(snapnum, self.snap_dir, min_mass)
            #Initialise arrays
            self.nHI=np.zeros(np.size(self.ind))
            self.MHI=np.zeros(np.size(self.ind))
            self.Mgas=np.zeros(np.size(self.ind))
            self.tot_found=np.zeros(np.size(self.ind))
            print "Found ",np.size(self.ind)," halos with > ",minpart,"particles"
            star=cold_gas.RahmatiRT(self.redshift, self.hubble)
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
                inH0 = star.get_code_rhoHI(bar)
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



