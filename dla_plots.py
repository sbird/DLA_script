# vim: set fileencoding=utf-8
"""
This is a module for making plots like those in Tescari & Viel, based on the data gathered in the Halohi module
Figures implemented:
    5,6,9,10-13
Possible but not implemented:
    14
        """

import halohi
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

acol="blue"
gcol="red"
rcol="black"
astyle="-"
gstyle="-."

class PrettyHalo(halohi.HaloHI):
    """
    Derived class with extra methods for plotting a pretty (high-resolution) picture of the grid around a halo.
    """

    def plot_pretty_halo(self,num=0):
        """
        Plots a pretty (high-resolution) picture of the grid around a halo.
        """
        #Plot a figure
        vmax=np.max(self.sub_nHI_grid[num])
        maxdist=self.maxdist
        plt.imshow(self.sub_nHI_grid[num],origin='lower',extent=(-maxdist,maxdist,-maxdist,maxdist),vmin=0,vmax=vmax)
        bar=plt.colorbar(use_gridspec=True)
        bar.set_label("log$_{10}$ N$_{HI}$ (cm$^{-2}$)")
        plt.xlabel("x (kpc/h)")
        plt.xlabel("y (kpc/h)")
        plt.tight_layout()
        plt.show()

    def plot_pretty_gas_halo(self,num=0):
        """
        Plots a pretty (high-resolution) picture of the grid around a halo.
        """
        #Plot a figure
        vmax=np.max(self.sub_gas_grid[num])
        maxdist=self.maxdist
        plt.imshow(self.sub_gas_grid[num],origin='lower',extent=(-maxdist,maxdist,-maxdist,maxdist),vmin=0,vmax=vmax)
        bar=plt.colorbar(use_gridspec=True)
        bar.set_label("log$_{10}$ N$_{H}$ (cm$^{-2}$)")
        plt.xlabel("x (kpc/h)")
        plt.xlabel("y (kpc/h)")
        plt.tight_layout()
        plt.show()


def plot_totalHI(base,snapnum,minpart=1000):
    """Make the plot of total neutral hydrogen density in a halo:
        Figure 9 of Tescari & Viel 2009"""
    #Get paths
    gdir=path.join(base,"Gadget")
    adir=path.join(base,"Arepo_ENERGY")
    #Load data
    atHI=halohi.TotalHaloHI(adir,snapnum,minpart)
    gtHI=halohi.TotalHaloHI(gdir,snapnum,minpart)
    #Plot.
    plt.loglog(gtHI.mass,gtHI.nHI,'o',color=gcol,label="Gadget")
    plt.loglog(atHI.mass,atHI.nHI,'o',color=acol,label="Arepo")
    #Axes
    plt.xlabel(r"Mass ($M_\odot$/h)")
    plt.ylabel("HI frac")
    plt.legend(loc=4)
    plt.xlim(1e9,5e12)
    plt.show()
    plt.tight_layout()
    return


class HaloHIPlots:
    """
    This class contains functions for plotting all the plots in
    Tescari and Viel which are derived from the grid of HI density around the halos.
    These are figs 10-13
    """
    def __init__(self,base,snapnum,minpart=10**5,ngrid=None,maxdist=100.,minplot=1e9, maxplot=5e12):
        #Get paths
        self.gdir=path.join(base,"Gadget")
        self.adir=path.join(base,"Arepo_ENERGY")
        #Get data
        self.ahalo=PrettyHalo(self.adir,snapnum,minpart,ngrid,maxdist)
        self.ahalo.save_file()
        self.ghalo=PrettyHalo(self.gdir,snapnum,minpart,ngrid,maxdist)
        self.ghalo.save_file()
        self.minplot=minplot
        self.maxplot=maxplot
        #Get the DLA redshift fit
        self.aDLAdz=halohi.DNdlaDz(self.ahalo.get_sigma_DLA(),self.ahalo.sub_mass,self.ahalo.redshift,self.ahalo.omegam,self.ahalo.omegal,self.ahalo.hubble)
        self.gDLAdz=halohi.DNdlaDz(self.ghalo.get_sigma_DLA(),self.ghalo.sub_mass,self.ghalo.redshift,self.ghalo.omegam,self.ghalo.omegal,self.ghalo.hubble)

    def plot_sigma_DLA(self):
        """Plot sigma_DLA against mass. Figure 10."""
        mass=np.logspace(np.log10(np.min(self.ahalo.sub_mass)),np.log10(np.max(self.ahalo.sub_mass)),num=100)
        alabel = r"Arepo: $\alpha=$"+str(np.round(self.aDLAdz.alpha,2))+" $\\beta=$"+str(np.round(self.aDLAdz.beta,2))
        glabel = r"Gadget: $\alpha=$"+str(np.round(self.gDLAdz.alpha,2))+" $\\beta=$"+str(np.round(self.gDLAdz.beta,2))
        plt.loglog(mass,self.aDLAdz.sigma_DLA_fit(mass),color=acol,label=alabel,ls=astyle)
        plt.loglog(mass,self.gDLAdz.sigma_DLA_fit(mass),color=gcol,label=glabel,ls=gstyle)
        #Axes
        plt.xlabel(r"Mass ($M_\odot$/h)")
        plt.ylabel(r"$\sigma_{DLA}$ (kpc$^2$/h$^2$)")
        plt.legend(loc=0)
        plt.loglog(self.ghalo.sub_mass,self.ghalo.get_sigma_DLA(),'s',color=gcol)
        plt.loglog(self.ahalo.sub_mass,self.ahalo.get_sigma_DLA(),'^',color=acol)
        plt.xlim(self.minplot,self.maxplot)
        #Fits
        plt.tight_layout()
        plt.show()

    def get_rel_sigma_DLA(self):
        """Get the change in sigma_DLA for a particular halo.
         and the mass of each halo averaged across arepo and gadget.
        """
        aDLA=self.ahalo.get_sigma_DLA()
        gDLA=self.ghalo.get_sigma_DLA()
        rDLA=np.empty(np.size(aDLA))
        rmass=np.empty(np.size(aDLA))
        for ii in xrange(0,np.size(aDLA)):
            aindex=self.ahalo.ind[0][ii]
            gg=np.where(self.ghalo.ind[0] == aindex)
            if np.size(gg) > 0 and aDLA[ii]+gDLA[gg] > 0:
                rDLA[ii] = 2*(aDLA[ii]-gDLA[gg])/(aDLA[ii]+gDLA[gg])
                rmass[ii]=0.5*(self.ahalo.sub_mass[ii]+self.ghalo.sub_mass[gg])
            else:
                rDLA[ii]=np.NaN
                rmass[ii]=np.NaN
        ind=np.where(np.isnan(rDLA) != True)
        return (rmass[ind],rDLA[ind])


    def plot_rel_sigma_DLA(self):
        """Plot sigma_DLA against mass. Figure 10."""
        (rmass,rDLA)=self.get_rel_sigma_DLA()
        plt.semilogx(rmass,rDLA,'o',color=rcol)
        plt.xlim(self.minplot,self.maxplot)
        #Axes
        plt.xlabel(r"Mass ($M_\odot$/h)")
        plt.ylabel(r"$\delta \sigma_\mathrm{DLA} / \sigma_\mathrm{DLA}$ (kpc$^2$/h$^2$)")
        #Fits
        plt.tight_layout()
        plt.show()

    def plot_dN_dla(self,Mmin=1e9,Mmax=1e13):
        """Plots dN_DLA/dz for the halos. Figure 11"""
        Mmax=np.min([Mmax,10**self.aDLAdz.log_mass_lim[1]])
        mass=np.logspace(np.log10(Mmin),np.log10(Mmax),num=100)
        aDLA_dz_tab = np.empty(np.size(mass))
        gDLA_dz_tab = np.empty(np.size(mass))
        for (i,m) in enumerate(mass):
            aDLA_dz_tab[i] = self.aDLAdz.get_N_DLA_dz(m)
            gDLA_dz_tab[i] = self.gDLAdz.get_N_DLA_dz(m)
        print "AREPO: alpha=",self.aDLAdz.alpha," beta=",self.aDLAdz.beta
        print "GADGET: alpha=",self.gDLAdz.alpha," beta=",self.gDLAdz.beta
        plt.loglog(mass,aDLA_dz_tab,color=acol,label="Arepo",ls=astyle)
        plt.loglog(mass,gDLA_dz_tab,color=gcol,label="Gadget",ls=gstyle)
        plt.xlabel(r"Mass ($M_\odot$/h)")
        plt.ylabel(r"$dN_{DLA} / dz (> M_\mathrm{tot})$")
        plt.legend(loc=3)
        plt.xlim(self.minplot,self.maxplot)
        plt.tight_layout()
        plt.show()

    def plot_column_density(self,minN=10,maxN=25.):
        """Plots the column density distribution function. Figures 12 and 13"""
        (aNHI,af_N)=self.ahalo.column_density_function(0.2,minN,maxN)
        (gNHI,gf_N)=self.ghalo.column_density_function(0.2,minN,maxN)
        plt.loglog(aNHI,af_N,color=acol, ls=astyle,label="Arepo")
        plt.loglog(aNHI,gf_N,color=gcol, ls=gstyle,label="Arepo / Gadget")
        #Make the ticks be less-dense
        #ax=plt.gca()
        #ax.xaxis.set_ticks(np.power(10.,np.arange(int(minN),int(maxN),2)))
        #ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(af_N[-1])),int(np.log10(af_N[0])),2)))
        plt.xlabel(r"$N_{HI} (\mathrm{cm}^{-2})$")
        plt.ylabel(r"$f(N) (\mathrm{cm}^2)$")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

    def plot_rel_column_density(self,minN=10,maxN=25.):
        """Plots the column density distribution function. Figures 12 and 13"""
        (aNHI,af_N)=self.ahalo.column_density_function(0.2,minN,maxN)
        (gNHI,gf_N)=self.ghalo.column_density_function(0.2,minN,maxN)
        plt.loglog(aNHI,af_N/gf_N,label="Arepo / Gadget",color=rcol)
        #Make the ticks be less-dense
        ax=plt.gca()
        ax.xaxis.set_ticks(np.power(10.,np.arange(int(minN),int(maxN),3)))
        #ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(af_N[-1])),int(np.log10(af_N[0])),2)))
        plt.xlabel(r"$N_{HI} (\mathrm{cm}^{-2})$")
        plt.ylabel(r"$ \delta f(N) (\mathrm{cm}^2)$")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

    def plot_halo_mass_func(self):
        """Plots the halo mass function as well as Sheth-Torman. Figure 5."""
        mass=np.logspace(np.log10(self.minplot),np.log10(self.maxplot),51)
        shdndm=[self.aDLAdz.halo_mass.dndm(mm) for mm in mass]
        adndm=np.empty(50)
        gdndm=np.empty(50)
        for ii in range(0,50):
            adndm[ii]=self.ahalo.get_dndm(mass[ii],mass[ii+1])
            gdndm[ii]=self.ghalo.get_dndm(mass[ii],mass[ii+1])
        plt.loglog(mass,shdndm,color="black",ls='--',label="Sheth-Tormen")
        plt.loglog(mass[0:-1],adndm,color=acol,ls=astyle,label="Arepo")
        plt.loglog(mass[0:-1],gdndm,color=gcol,ls=gstyle,label="Gadget")
        #Make the ticks be less-dense
        ax=plt.gca()
        ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(shdndm[-1])),int(np.log10(shdndm[0])),2)))

        plt.ylabel(r"dn/dM (h$^4$ $M^{-1}_\odot$ Mpc$^{-3}$)")
        plt.xlabel(r"Mass ($M_\odot$/h)")
        plt.legend(loc=0)
        plt.xlim(self.minplot,self.maxplot)
        plt.tight_layout()
        plt.show()
