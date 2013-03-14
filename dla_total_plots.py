"""Module containing classes for plotting functions associated with the TotalHaloHi class and the VelocityHI class"""

import totalhalohi
import velhi
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import scipy
from dla_plots import tight_layout_wrapper

gcol="blue"
acol="red"

class PrettyTotalHI(totalhalohi.TotalHaloHI):
    """Derived class for plotting total nHI frac and total nHI mass
    against halo mass"""
    def plot_totalHI(self,color="black",label=""):
        """Make the plot of total neutral hydrogen density in a halo:
            Figure 9 of Tescari & Viel 2009"""
        #Plot.
        plt.loglog(self.mass,self.nHI,'o',color=color,label=label)
        #Axes
        plt.xlabel(r"Mass ($M_\odot$ h$^{-1}$)")
        plt.ylabel("HI frac")
        plt.xlim(1e9,5e12)

    def plot_MHI(self,color="black",label=""):
        """Total M_HI vs M_halo"""
        #Plot.
        plt.loglog(self.mass,self.MHI,'o',color=color)
        #Make a best-fit curve.
        ind=np.where(self.MHI > 0.)
        logmass=np.log10(self.mass[ind])-12
        loggas=np.log10(self.MHI[ind])
        ind2=np.where(logmass > -2)
        (alpha,beta)=scipy.polyfit(logmass[ind2],loggas[ind2],1)
        mass_bins=np.logspace(np.log10(np.min(self.mass)),np.log10(np.max(self.mass)),num=100)
        fit= 10**(alpha*(np.log10(mass_bins)-12)+beta)
        plt.loglog(mass_bins,fit, color=color,label=label+r"$\alpha$="+str(np.round(alpha,2))+r" $\beta$ = "+str(np.round(beta,2)))
        #Axes
        plt.xlabel(r"Mass ($M_\odot$ h$^{-1}$)")
        plt.ylabel(r"Mass$_{HI}$ ($M_\odot$ h$^{-1}$)")
        plt.xlim(1e9,5e12)

    def plot_gas(self,color="black",label=""):
        """Total M_gas vs M_halo"""
        #Plot.
        plt.loglog(self.mass,self.Mgas,'o',color=color)
        #Make a best-fit curve.
        ind=np.where(self.Mgas > 0.)
        logmass=np.log10(self.mass[ind])-12
        loggas=np.log10(self.Mgas[ind])
        ind2=np.where(logmass > -2)
        (alpha,beta)=scipy.polyfit(logmass[ind2],loggas[ind2],1)
        mass_bins=np.logspace(np.log10(np.min(self.mass)),np.log10(np.max(self.mass)),num=100)
        fit= 10**(alpha*(np.log10(mass_bins)-12)+beta)
        plt.loglog(mass_bins,fit, color=color,label=label+r"$\alpha$="+str(np.round(alpha,2))+r" $\beta$ = "+str(np.round(beta,2)))
        #Axes
        plt.xlabel(r"Mass ($M_\odot$ h$^{-1}$)")
        plt.ylabel(r"Mass$_{gas}$ ($M_\odot$ h$^{-1}$)")
        plt.xlim(1e9,5e12)


class TotalHIPlots:
    """Class for plotting functions from PrettyHaloHI"""
    def __init__(self,base,snapnum,minpart=400):
        #Get paths
        gdir=path.join(base,"Gadget")
        adir=path.join(base,"Arepo_ENERGY")
        #Load data
        self.atHI=PrettyTotalHI(adir,snapnum,minpart)
        self.atHI.save_file()
        self.gtHI=PrettyTotalHI(gdir,snapnum,minpart)
        self.gtHI.save_file()

    def plot_totalHI(self):
        """Make the plot of total neutral hydrogen density in a halo:
            Figure 9 of Tescari & Viel 2009"""
        #Plot.
        self.gtHI.plot_totalHI(color=gcol,label="Gadget")
        self.atHI.plot_totalHI(color=acol,label="Arepo")
        #Axes
        plt.legend(loc=4)
        tight_layout_wrapper()
        plt.show()

    def plot_MHI(self):
        """Make the plot of total neutral hydrogen mass in a halo:
            Figure 9 of Tescari & Viel 2009"""
        #Plot.
        self.gtHI.plot_MHI(color=gcol,label="Gadget")
        self.atHI.plot_MHI(color=acol,label="Arepo")
        #Axes
        plt.legend(loc=0)
        tight_layout_wrapper()
        plt.show()

    def plot_gas(self):
        """Plot total gas mass in a halo"""
        #Plot.
        self.gtHI.plot_gas(color=gcol,label="Gadget")
        self.atHI.plot_gas(color=acol,label="Arepo")
        #Axes
        plt.legend(loc=0)
        tight_layout_wrapper()
        plt.show()


class PrettyVelocity(velhi.VelocityHI):
    """
    Make a velocity plot
    """
    def __init__(self,snap_dir,snapnum,minpart,reload_file=False,skip_grid=None,savefile=None):
        velhi.VelocityHI.__init__(self,snap_dir,snapnum,minpart,reload_file=reload_file,skip_grid=None,savefile=savefile)

    def radial_log(self,x,y,cut=1e25):
        """If we have x and y st. x+iy = r e^iθ, find x' and y' s.t. x'+iy' = log(r) e^iθ"""
        r = np.sqrt(x**2+y**2)
        ind = np.where(r > cut)
        sc=np.ones(np.shape(r))
        sc[ind] = cut/r[ind]
        return (x*sc, y*sc)

    def plot_velocity_map(self,num=0,scale=1e32,cut=1e25):
        """Plot the velocity map around a halo"""
        (x,y) = self.radial_log(self.sub_nHI_grid[num],self.sub_gas_grid[num],cut=cut)
        r = np.sqrt(x**2+y**2)
        plt.quiver(x,y,r,scale=scale,scale_units='xy',)

