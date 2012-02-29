# vim: set fileencoding=utf-8
"""
This is a module for making plots like those in Tescari & Viel, based on the data gathered in the Halohi module
Figures implemented:
    6,9,10-13
Possible but not implemented:
    5,14
        """

import halohi
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

acol="blue"
gcol="red"
astyle="-"
gstyle="-."

def plot_pretty_halo(base,snapnum,halo=50,ngrid=255,maxdist=100.):
    """
    Plots a pretty (high-resolution) picture of the grid around a halo.
    Like Figure 6 of Tescari & Viel
    """
    #Get paths
    gdir=path.join(base,"Gadget")
    adir=path.join(base,"Arepo_ENERGY")
    #Get data
    if np.size(halo) == 0:
        raise ValueError("Must specify a list of halos to pretty plot!")
    if np.size(halo) ==1:
        halo_list=[halo,]
    else:
        halo_list=list(halo)
    ahalo=halohi.HaloHI(adir,snapnum,100,ngrid,maxdist,halo_list)
    #ahalo.save_file()
    ghalo=halohi.HaloHI(gdir,snapnum,100,ngrid,maxdist,halo_list)
    #ghalo.save_file()
    #Plot a figure
    plt.imshow(ahalo.sub_nHI_grid,origin='lower',extent=(-maxdist,maxdist,-maxdist,maxdist),vmin=0,vmax=55)
    bar=plt.colorbar()
    bar.set_label("log$_{10}$ N$_{HI}$ (cm$^{-2}$)")
    plt.xlabel("x (kpc/h)")
    plt.xlabel("y (kpc/h)")
    plt.title("Arepo")
    plt.show()
    #Plot another figure
    #Plot a figure
    plt.imshow(ghalo.sub_nHI_grid,origin='lower',extent=(-maxdist,maxdist,-maxdist,maxdist),vmin=0,vmax=55)
    bar=plt.colorbar()
    bar.set_label("log$_{10}$ N$_{HI}$ (cm$^{-2}$)")
    plt.xlabel("x (kpc/h)")
    plt.xlabel("y (kpc/h)")
    plt.title("Gadget")
    plt.show()


def plot_totalHI(base,snapnum,minpart=1000):
    """Make the plot of total neutral hydrogen density in a halo:
        Figure 9 of Tescari & Viel 2009"""
    #Get paths
    gdir=path.join(base,"Gadget")
    adir=path.join(base,"Arepo_ENERGY")
    #Load data
    atHI=halohi.TotalHaloHI(adir,snapnum,minpart)
    atHI.save_file()
    gtHI=halohi.TotalHaloHI(gdir,snapnum,minpart)
    gtHI.save_file()
    #Plot.
    plt.loglog(atHI.mass,atHI.nHI,'o',color=acol,label="Arepo")
    plt.loglog(gtHI.mass,gtHI.nHI,'o',color=gcol,label="Gadget")
    #Axes
    plt.xlabel(r"Mass ($M_\odot$)")
    plt.ylabel("HI frac")
    plt.legend(loc=4)
    plt.show()
    return


class HaloHIPlots:
    """
    This class contains functions for plotting all the plots in
    Tescari and Viel which are derived from the grid of HI density around the halos.
    These are figs 10-13
    """
    def __init__(self,base,snapnum,minpart=10**5,ngrid=33,maxdist=100.):
        #Get paths
        self.gdir=path.join(base,"Gadget")
        self.adir=path.join(base,"Arepo_ENERGY")
        #Get data
        self.ahalo=halohi.HaloHI(self.adir,snapnum,minpart,ngrid,maxdist)
        self.ahalo.save_file()
        self.ghalo=halohi.HaloHI(self.gdir,snapnum,minpart,ngrid,maxdist)
        self.ghalo.save_file()
        #Get the DLA redshift fit
        self.aDLAdz=halohi.DNdlaDz(self.ahalo.get_sigma_DLA(),self.ahalo.sub_mass,self.ahalo.redshift,self.ahalo.omegam,self.ahalo.omegal,self.ahalo.hubble)
        self.gDLAdz=halohi.DNdlaDz(self.ghalo.get_sigma_DLA(),self.ghalo.sub_mass,self.ghalo.redshift,self.ghalo.omegam,self.ghalo.omegal,self.ghalo.hubble)

    def plot_sigma_DLA(self):
        """Plot sigma_DLA against mass. Figure 10."""
        plt.loglog(self.ahalo.sub_mass,self.ahalo.get_sigma_DLA(),'^',legend="Arepo",color=acol)
        plt.loglog(self.ghalo.sub_mass,self.ghalo.get_sigma_DLA(),'s',legend="Gadget",color=gcol)
        #Fits
        mass=np.logspace(np.min(self.ahalo.sub_mass),np.max(self.ahalo.sub_mass),num=100)
        plt.loglog(mass,self.aDLAdz.sigma_DLA_fit(mass),color=acol,legend="Arepo",ls=astyle)
        plt.loglog(mass,self.gDLAdz.sigma_DLA_fit(mass),color=gcol,legend="Gadget",ls=gstyle)
        #Axes
        plt.xlabel(r"Mass ($M_\odot$)")
        plt.ylabel(r"$\sigma_{DLA}$ ($(kpc/h)^2$)")
        plt.legend(loc=2)
        plt.show()

    def plot_dN_dla(self,Mmin=1e9,Mmax=1e13):
        """Plots dN_DLA/dz fro the halos. Figure 11"""
        mass=np.logspace(Mmin,Mmax,num=100)
        plt.loglog(mass,self.aDLAdz.get_N_DLA_dz(mass),color=acol,legend="Arepo",ls=astyle)
        plt.loglog(mass,self.gDLAdz.get_N_DLA_dz(mass),color=gcol,legend="Gadget",ls=gstyle)
        plt.xlabel(r"Mass ($M_\odot$)")
        plt.ylabel(r"$dN_{DLA} / dz (> M_\mathrm{tot})$")
        plt.legend(loc=3)
        plt.show()

    def plot_column_density(self):
        """Plots the column density distribution function. Figures 12 and 13"""
        (aNHI,af_N)=self.ahalo.column_density_function()
        (gNHI,gf_N)=self.ghalo.column_density_function()
        plt.loglog(aNHI,af_N,color=acol,ls=astyle,legend="Arepo")
        plt.loglog(gNHI,gf_N,color=gcol,ls=gstyle,legend="Gadget")
        plt.xlabel(r"$N_{HI} (\mathrm{cm}^{-2})$")
        plt.ylabel(r"$f(N) (\mathrm{cm}^2)$")
        plt.legend(loc=3)
        plt.show()

