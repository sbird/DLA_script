import numpy as np
import math
import matplotlib.pyplot as plt
import os.path as path
import sys
import re
import readsubf
import hdfsim
import cold_gas
import halo_mass_function
import fieldize
import scipy
import scipy.interpolate as interp
import scipy.integrate as integ
import scipy.weave
# cat = readsubf.subfind_catalog("./m_10002_h_94_501_z3_csf/",63,masstab=True)
# print cat.nsubs
# print "largest halo x position = ",cat.sub_pos[0][0] 


class spectra:
    def __init__(self,file,los_table="",nlos=16000,nbins=1024,no_header=1):
        self.nlos=nlos
        self.nbins=nbins
        self.read_spectra(file,no_header)
        if los_table != "":
            self.los_table=np.loadtxt(los_table)
        return

    def read_spectra(self,file,no_header=1):
        fd=open(file,'rb')
        self.zz=np.fromfile(fd,dtype=np.float64,count=1)
        if not no_header:
            self.box=np.fromfile(fd,dtype=np.float64,count=1)
            #This is an integer
            head=np.fromfile(fd,dtype=np.int32,count=2)
            self.nbins=head[0]
            self.nlos=head[1]
            #Pad to 128 bytes
            fd.seek(128,0)
        size=self.nbins*self.nlos
        #Density of hydrogen
        rho_H=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        #neutral hydrogen fraction
        rho_HI=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        self.n_HI=np.nansum(rho_HI/rho_H,axis=1)
        #temperature
        #temp_HI=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        #velocity
        #vel_HI=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        #optical depth
        #tau_HI=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        return

    def get_los(self,axis):
        ind=np.where(self.los_table[:,0] == axis)
        axes=np.where(np.array([axis,1,2,3]) != axis)
        los=np.empty((np.size(self.los_table[ind,axes[0][0]]),2))
        los[:,0]=self.los_table[ind,axes[0][0]]
        los[:,1]=self.los_table[ind,axes[0][1]]
        return (ind,los)

    def get_int_hi(self,axis=1):
        (ind,pos)=self.get_los(axis)
        sn_HI=self.n_HI[ind]
        return sn_HI

    def get_spec_pos(self,axis=1):
        (ind,pos)=self.get_los(axis)
        return (pos[:,0],pos[:,1])

    def plot_int_hi(self,axis,vmax=0):
        nHI=self.grid_int_hi(axis)
        if vmax ==0:
            plt.imshow(nHI.map,origin='lower',extent=nHI.get_lims(),aspect='auto')
        else:
            plt.imshow(nHI.map,origin='lower',extent=nHI.get_lims(),aspect='auto',vmax=vmax,vmin=0)
        plt.colorbar()
    
    def grid_int_hi(self,axis):
        n_HI=self.get_int_hi(axis)
        (x,y)=self.get_spec_pos(axis)
        grid_n_HI=quant_grid((x,y),n_HI,nbins=math.sqrt(self.nlos/3)/2)
        return grid_n_HI

class quant_grid:
    def __init__(self,pos,mass, box=20.,nbins=150):
        self.map = np.zeros([nbins,nbins]) 
        count = np.zeros([nbins,nbins]) 
        self.box=box
        self.x=np.array(np.shape(self.map))
        ind=self.co_ind(pos)
        for i in range(0,np.shape(ind)[1]):
            self.map[ind[0][i],ind[1][i]]+=mass[i]
            count[ind[0][i],ind[1][i]]+=1
        #Note this sets gridpoints with no spectra to Nan
        self.map/=count

    def co_ind(self,xx):
        x=np.array(xx)
        x[0]=np.array(xx[0])/self.box*(self.x[0]-1)
        x[1]=np.array(xx[1])/self.box*(self.x[1]-1)
        return np.array(np.around(x),dtype=int)

    def get_lims(self):
        return(0,self.box,0,self.box)

def gen_los_table(filename,nbins,box=20.):
    los_table=np.empty([3*nbins**2,4])
    sc=(1.*box)/nbins
    for j in range(0,nbins):
        for i in range(0,nbins):
            los_table[nbins*j+i,:]=[1,0,i*sc,j*sc]
    for j in range(0,nbins):
        for i in range(0,nbins):
            los_table[nbins**2+nbins*j+i,:]=[2,i*sc,0,j*sc]
    for j in range(0,nbins):
        for i in range(0,nbins):
            los_table[2*nbins**2+nbins*j+i,:]=[3,i*sc,j*sc,0]
    np.savetxt(filename,los_table,fmt="%d %.3e %.3e %.3e")

class subfind(readsubf.subfind_catalog):
    def __init__(self,dir,snapnum):
        self.dir=dir
        self.snapnum=snapnum
        readsubf.subfind_catalog.__init__(self,dir,snapnum,masstab=True,long_ids=True)

    def gen_halo_los_table(self, minmass=3e8):
        #Look at above-average mass halos only
        ind=np.where(self.sub_mass > minmass/1e10)
        nsubs=np.size(ind)
        #Make table of sightlines, one going through 
        #the center of each halo in each direction
        los_table=np.empty([3*nsubs,4])
        #x-axis
        los_table[0:nsubs,1:4]=self.sub_pos[ind]/1000
        los_table[0:nsubs,0]=1
        #y-axis
        los_table[nsubs:2*nsubs,1:4]=self.sub_pos[ind]/1000
        los_table[nsubs:2*nsubs,0]=2
        #z-axis
        los_table[2*nsubs:3*nsubs,1:4]=self.sub_pos[ind]/1000
        los_table[2*nsubs:3*nsubs,0]=3
        np.savetxt(path.join(self.dir,"los_"+str(self.snapnum)+".txt"),los_table,fmt="%d %.3e %.3e %.3e")
    
    def gen_halo_table(self,  minmass=3e8):
        #Look at above-average mass halos only
        ind=np.where(self.sub_mass > minmass/1e10)
        nsubs=np.size(ind)
        #Make table of sightlines, one going through
        #the center of each halo in each direction
        np.savetxt(path.join(self.dir,"halo_"+str(self.snapnum)+".txt"),self.sub_pos[ind]/1000,fmt="%.3e %.3e %.3e")

    def get_halo_mass(self,  minmass=3e8):
        #Look at above-average mass halos only
        ind=np.where(self.sub_mass > minmass/1e10)
        #Assume code units where one mass unit is 1e10 solar masses
        return self.sub_mass[ind]*1e10

class total_halo_HI:
    """Find the average HI fraction in a halo
    This is like Figure 9 of Tescari & Viel"""
    def __init__(self,dir,snapnum,minpart=1000):
        #f np > 1.4.0, we have in1d
        if not re.match("1\.[4-9]",np.version.version):
            print "Need numpy 1.4 for in1d: without it this is unfeasibly slow"
        #Get halo catalog
        subs=readsubf.subfind_catalog(dir,snapnum,masstab=True,long_ids=True)
        #Get list of halos resolved with > minpart particles
        ind=np.where(subs.sub_len > minpart)
        self.nHI=np.zeros(np.size(ind))
        self.tot_found=np.zeros(np.size(ind))
        print "Found ",np.size(ind)," halos with > ",minpart,"particles"
        #Get particle ids for each subhalo
        sub_ids=[readsubf.subf_ids(dir,snapnum,np.sum(subs.sub_len[0:i]),subs.sub_len[i],long_ids=True).SubIDs for i in np.ravel(ind)]
        all_sub_ids=np.concatenate(sub_ids)
        print "Got particle id lists"
        #Now find the average HI for each halo
        for fnum in range(0,500):
            try:
                f=hdfsim.get_file(snapnum,dir,fnum)
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

class halo_HI:
    def __init__(self,snap_dir,snapnum,minpart=10**4,ngrid=32,maxdist=100.):
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
        self.minpart=minpart
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        #f np > 1.4.0, we have in1d
        if not re.match("1\.[4-9]",np.version.version):
            print "Need numpy 1.4 for in1d: without it this is unfeasibly slow"
        #Get halo catalog
        subs=readsubf.subfind_catalog(self.snap_dir,snapnum,masstab=True,long_ids=True)
        #Get list of halos resolved with > minpart particles
        ind=np.where(subs.sub_len > minpart)
        self.nHI=np.zeros(np.size(ind))
        self.tot_found=np.zeros(np.size(ind))
        nhalo=np.size(ind)
        print "Found ",nhalo," halos with > ",minpart,"particles"
        #Get particle center of mass 
        self.sub_cofm=np.array(subs.sub_pos[ind])
        #halo masses
        self.sub_mass=np.array(subs.sub_mass[ind])
        del subs
        #Grid to put paticles on
        f=hdfsim.get_file(snapnum,self.snap_dir,0)
        self.redshift=f["Header"].attrs["Redshift"]
        f.close()
        self.set_nHI_grid(ngrid,maxdist)
        return


    def set_nHI_grid(self,ngrid=32,maxdist=100.):
        """Set up the grid around each halo where the HI is calculated.
            ngrid - Size of grid to store values on
            maxdist - Maximum extent of grid in kpc.
        """
        self.ngrid=ngrid
        self.maxdist=maxdist
        #Internal gadget mass unit: 1e10 M_sun in g
        UnitMass_in_g=1.989e43
        UnitLength_in_cm=3.085678e21
        self.sub_nHI_grid=[np.zeros((self.ngrid+1,self.ngrid+1)) for i in self.sub_cofm]
        #Now grid the HI for each halo
        for fnum in xrange(0,500):
            try:
                f=hdfsim.get_file(snapnum,self.snap_dir,fnum)
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
            coords=[((ppos- self.sub_cofm[idx])/(2*self.maxdist)+0.5)*self.ngrid for idx,ppos in enumerate(poslist)]
            #NH0
            rhoH0 = [irhoH0[ind] for ind in near_halo]
            [fieldize.ngp(coords[i],rhoH0[i],self.sub_nHI_grid[i]) for i in xrange(0,nhalo)]
        #Linear dimension of each cell in cm
        epsilon=2.*self.maxdist/(self.ngrid+1)*UnitLength_in_cm
        self.sub_nHI_grid=[g*epsilon/(1+self.redshift)**2 for g in self.sub_nHI_grid]
        for ii,grid in enumerate(self.sub_nHI_grid):
            ind=np.where(grid > 0)
            grid[ind]=np.log(grid[ind])
            self.sub_nHI_grid[ii]=grid
        return

    def get_sigma_DLA(self):
        """Get the DLA cross-section from the neutral hydrogen column densities found in this class.
        This is defined as the area of all the cells with column density above 10^20.3 cm^-2.
        Returns result in (kpc/h)^2."""
        cell_area=4*self.maxdist**2/(self.ngrid+1)**2
        self.sigma_DLA = [ np.sum(grid[np.where(grid > 20.3)])*cell_area for grid in self.sub_nHI_grid]
        return

class d_N_DLA_dz:
    def __init__(self, sigma_DLA,halo_mass, redshift,Omega_M=0.27, Omega_L = 0.73, hubble=0.7):
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
        self.halo_mass=halo_mass_function.halo_mass_function(redshift,omega_m=Omega_M, omega_l=Omega_L, hubble=hubble,log_mass_lim=self.log_mass_lim)

    def sigma_DLA_fit(self,M):
        return np.exp(self.alpha*(np.log(M)-12)+self.beta)


    def drdz(self,zz):
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
        M=np.exp(logM)
        return self.sigma_DLA_fit(M)*self.halo_mass.dndm(M)*M
