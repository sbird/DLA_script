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
import scipy
import os.path as path
import matplotlib.pyplot as plt

acol="blue"
gcol="red"
rcol="black"
astyle="-"
gstyle="--"

class PrettyHalo(halohi.HaloHI):
    """
    Derived class with extra methods for plotting a pretty (high-resolution) picture of the grid around a halo.
    """

    def plot_pretty_halo(self,num=0):
        """
        Plots a pretty (high-resolution) picture of the grid around a halo.
        """
        #Plot a figure
        vmax=np.max(self.sub_nHI_grid[num,:,:])
        maxdist=self.maxdist
        plt.imshow(self.sub_nHI_grid[num,:,:],origin='lower',extent=(-maxdist,maxdist,-maxdist,maxdist),vmin=0,vmax=vmax)
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
        vmax=np.max(self.sub_gas_grid[num,:,:])
        maxdist=self.maxdist
        plt.imshow(self.sub_gas_grid[num,:,:],origin='lower',extent=(-maxdist,maxdist,-maxdist,maxdist),vmin=0,vmax=vmax)
        bar=plt.colorbar(use_gridspec=True)
        bar.set_label("log$_{10}$ N$_{H}$ (cm$^{-2}$)")
        plt.xlabel("x (kpc/h)")
        plt.xlabel("y (kpc/h)")
        plt.tight_layout()
        plt.show()

    def plot_radial_profile(self,minM=2e11,maxM=1e12,minR=0,maxR=100.):
        """Plots the radial density of neutral hydrogen (and possibly gas) for a given halo,
        stacking several halo profiles together."""
        Rbins=np.linspace(minR,maxR,20)
        try:
            aRprof=[self.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1]) for i in xrange(0,np.size(Rbins)-1)]
            plt.semilogy(Rbins[0:-1],aRprof,color=acol, ls=astyle,label="HI")
            #If we didn't load the HI grid this time
        except AttributeError:
            pass
        #Gas profiles
        try:
            agRprof=[self.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1],True) for i in xrange(0,np.size(Rbins)-1)]
            plt.semilogy(Rbins[0:-1],agRprof,color="brown", ls=astyle,label="Gas")
        except AttributeError:
            pass
        plt.xlabel(r"R (kpc/h)")
        plt.ylabel(r"Density $N_{HI}$ (kpc$^{-1}$)")
        plt.legend(loc=1)
        plt.tight_layout()
        plt.show()

    def plot_gas_vs_halo_mass(self,label="",color="black"):
        """Plot Gas mass vs total halo mass"""
        #Plot.
        plt.loglog(self.sub_mass,self.sub_gas_mass,'o',color=color)
        #Make a best-fit curve.
        ind=np.where(self.sub_gas_mass > 0.)
        logmass=np.log10(self.sub_mass[ind])-12
        loggas=np.log10(self.sub_gas_mass[ind])
        (alpha,beta)=scipy.polyfit(logmass,loggas,1)
        mass_bins=np.logspace(np.log10(np.min(self.sub_mass)),np.log10(np.max(self.sub_mass)),num=100)
        gas_fit= 10**(alpha*(np.log10(mass_bins)-12)+beta)
        plt.loglog(mass_bins,gas_fit, color=color,label=label+r" $\alpha$="+str(np.round(alpha,2))+r" $\beta$ = "+str(np.round(beta,2)))
        #Axes
        plt.xlabel(r"Mass ($M_\odot$/h)")
        plt.ylabel("Gas Mass ($M_\odot$/h)")
        plt.xlim(1e9,5e12)
        plt.tight_layout()
        plt.show()
        return


class PrettyBox(halohi.BoxHI,PrettyHalo):
    """
    As above but for the whole box grid
    """
    def __init__(self,snap_dir,snapnum,ngrid=None,reload_file=False,skip_grid=None,savefile=None):
        halohi.BoxHI.__init__(self,snap_dir,snapnum,ngrid=None,reload_file=False,skip_grid=None,savefile=None)


class PrettyTotalHI(halohi.TotalHaloHI):
    """Derived class for plotting total nHI frac and total nHI mass
    against halo mass"""
    def plot_totalHI(self,color="black",label=""):
        """Make the plot of total neutral hydrogen density in a halo:
            Figure 9 of Tescari & Viel 2009"""
        #Plot.
        plt.loglog(self.mass,self.nHI,'o',color=color,label=label)
        #Axes
        plt.xlabel(r"Mass ($M_\odot$/h)")
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
        (alpha,beta)=scipy.polyfit(logmass,loggas,1)
        mass_bins=np.logspace(np.log10(np.min(self.mass)),np.log10(np.max(self.mass)),num=100)
        fit= 10**(alpha*(np.log10(mass_bins)-12)+beta)
        plt.loglog(mass_bins,fit, color=color,label=label+r"$\alpha$="+str(np.round(alpha,2))+r" $\beta$ = "+str(np.round(beta,2)))
        #Axes
        plt.xlabel(r"Mass ($M_\odot$/h)")
        plt.ylabel(r"Mass$_{HI}$ ($M_\odot$/h)")
        plt.xlim(1e9,5e12)

class TotalHIPlots:
    """Class for plotting functions from PrettyHaloHI"""
    def __init__(self,base,snapnum,minpart=1000):
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
        plt.tight_layout()
        plt.show()

    def plot_MHI(self):
        """Make the plot of total neutral hydrogen density in a halo:
            Figure 9 of Tescari & Viel 2009"""
        #Plot.
        self.gtHI.plot_MHI(color=gcol,label="Gadget")
        self.atHI.plot_MHI(color=acol,label="Arepo")
        #Axes
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()


class HaloHIPlots:
    """
    This class contains functions for plotting all the plots in
    Tescari and Viel which are derived from the grid of HI density around the halos.
    These are figs 10-13
    """
    def __init__(self,base,snapnum,minpart=10**5,ngrid=None,maxdist=100.,minplot=1e9, maxplot=5e12,reload_file=False,skip_grid=None):
        #Get paths
        self.gdir=path.join(base,"Gadget")
        self.adir=path.join(base,"Arepo_ENERGY")
        #Get data
        self.ahalo=PrettyHalo(self.adir,snapnum,minpart,ngrid,maxdist,reload_file=reload_file,skip_grid=skip_grid)
#         self.ahalo.save_file()
        self.ghalo=PrettyHalo(self.gdir,snapnum,minpart,ngrid,maxdist,reload_file=reload_file,skip_grid=skip_grid)
#         self.ghalo.save_file()
        self.minplot=minplot
        self.maxplot=maxplot
        #Get the DLA redshift fit
#         self.aDLAdz=halohi.DNdlaDz(self.ahalo.get_sigma_DLA(),self.ahalo.sub_mass,self.ahalo.redshift,self.ahalo.omegam,self.ahalo.omegal,self.ahalo.hubble)
#         self.gDLAdz=halohi.DNdlaDz(self.ghalo.get_sigma_DLA(),self.ghalo.sub_mass,self.ghalo.redshift,self.ghalo.omegam,self.ghalo.omegal,self.ghalo.hubble)

    def plot_sigma_DLA(self, DLA_cut=20.3):
        """Plot sigma_DLA against mass. Figure 10."""
        mass=np.logspace(np.log10(np.min(self.ahalo.sub_mass)),np.log10(np.max(self.ahalo.sub_mass)),num=100)
        asfit=self.ahalo.sigma_DLA_fit(mass,DLA_cut)
        gsfit=self.ghalo.sigma_DLA_fit(mass,DLA_cut)
        alabel = r"Arepo: $\alpha=$"+str(np.round(self.ahalo.alpha,2))+" $\\beta=$"+str(np.round(self.ahalo.beta,2))
        glabel = r"Gadget: $\alpha=$"+str(np.round(self.ghalo.alpha,2))+" $\\beta=$"+str(np.round(self.ghalo.beta,2))
        plt.loglog(mass,asfit,color=acol,label=alabel,ls=astyle)
        plt.loglog(mass,gsfit,color=gcol,label=glabel,ls=gstyle)
        #Axes
        plt.xlabel(r"Mass ($M_\odot$/h)")
        plt.ylabel(r"$\sigma_{DLA}$ (kpc$^2$/h$^2$) DLA is N > "+str(DLA_cut))
        plt.legend(loc=0)
        plt.loglog(self.ghalo.sub_mass,self.ghalo.get_sigma_DLA(DLA_cut),'s',color=gcol)
        plt.loglog(self.ahalo.sub_mass,self.ahalo.get_sigma_DLA(DLA_cut),'^',color=acol)
        plt.xlim(self.minplot,self.maxplot)
        #Fits
        plt.tight_layout()
        plt.show()

    def plot_sigma_DLA_gas(self, DLA_cut=20.3):
        """Plot sigma_DLA against gas mass. """
        gas_mass=np.logspace(np.log10(np.min(self.ahalo.sub_gas_mass)),np.log10(np.max(self.ahalo.sub_gas_mass)),num=100)
        asfit=self.ahalo.sigma_DLA_fit_gas(gas_mass,DLA_cut)
        gsfit=self.ghalo.sigma_DLA_fit_gas(gas_mass,DLA_cut)
        alabel = r"Arepo: $\alpha=$"+str(np.round(self.ahalo.alpha_g,2))+" $\\beta=$"+str(np.round(self.ahalo.beta_g,2))
        glabel = r"Gadget: $\alpha=$"+str(np.round(self.ghalo.alpha_g,2))+" $\\beta=$"+str(np.round(self.ghalo.beta_g,2))
        plt.loglog(gas_mass,asfit,color=acol,label=alabel,ls=astyle)
        plt.loglog(gas_mass,gsfit,color=gcol,label=glabel,ls=gstyle)
        #Axes
        plt.xlabel(r"Mass Hydrogen ($M_\odot$/h)")
        plt.ylabel(r"$\sigma_{DLA}$ (kpc$^2$/h$^2$) DLA is N > "+str(DLA_cut))
        plt.legend(loc=0)
        plt.loglog(self.ghalo.sub_gas_mass,self.ghalo.get_sigma_DLA(DLA_cut),'s',color=gcol)
        plt.loglog(self.ahalo.sub_gas_mass,self.ahalo.get_sigma_DLA(DLA_cut),'^',color=acol)
        plt.xlim(self.minplot/100.,self.maxplot/100.)
        #Fits
        plt.tight_layout()
        plt.show()

    def plot_sigma_DLA_nHI(self, DLA_cut=20.3):
        """Plot sigma_DLA against HI mass."""
        #Get MHI
        athi=PrettyTotalHI(self.adir,self.ahalo.snapnum,self.ahalo.minpart)
        gthi=PrettyTotalHI(self.gdir,self.ahalo.snapnum,self.ahalo.minpart)
        anHI_mass = np.array([athi.get_hi_mass(mass) for mass in self.ahalo.sub_mass])
        gnHI_mass = np.array([gthi.get_hi_mass(mass) for mass in self.ghalo.sub_mass])
        #Filter nan
        ind = np.where(anHI_mass > 0)
        anHI_mass=anHI_mass[ind]
        asigDLA=self.ahalo.get_sigma_DLA(DLA_cut)[ind]
        ind = np.where(gnHI_mass > 0)
        gnHI_mass=gnHI_mass[ind]
        gsigDLA=self.ghalo.get_sigma_DLA(DLA_cut)[ind]
        #Plot
        plt.loglog(gnHI_mass,gsigDLA,'s',color=gcol)
        plt.loglog(anHI_mass,asigDLA,'^',color=acol)
        hi_mass=np.logspace(np.log10(np.min(anHI_mass)),np.log10(np.max(anHI_mass)),num=100)
        #Get fit parameters
        ind=np.where(np.logical_and(asigDLA > 0.,anHI_mass > 0.))
        logmass=np.log10(anHI_mass[ind])-12
        logsigma=np.log10(asigDLA[ind])
        if np.size(logsigma) != 0:
            (alpha_a,beta_a)=scipy.polyfit(logmass,logsigma,1)
            alabel = r"Arepo: $\alpha=$"+str(np.round(alpha_a,2))+" $\\beta=$"+str(np.round(beta_a,2))
            asfit=10**(alpha_a*(np.log10(hi_mass)-12)+beta_a)
        ind=np.where(np.logical_and(gsigDLA > 0.,gnHI_mass > 0.))
        logmass=np.log10(gnHI_mass[ind])-12
        logsigma=np.log10(gsigDLA[ind])
        if np.size(logsigma) != 0:
            (alpha_g,beta_g)=scipy.polyfit(logmass,logsigma,1)
            glabel = r"Gadget: $\alpha=$"+str(np.round(alpha_g,2))+" $\\beta=$"+str(np.round(beta_g,2))
            gsfit=10**(alpha_g*(np.log10(hi_mass)-12)+beta_g)
            #Plot
        plt.loglog(hi_mass,gsfit,color=gcol,label=glabel,ls=gstyle)
        plt.loglog(hi_mass,asfit,color=acol,label=alabel,ls=astyle)
        #Axes
        plt.xlabel(r"Mass Hydrogen ($M_\odot$/h)")
        plt.ylabel(r"$\sigma_{DLA}$ (kpc$^2$/h$^2$) DLA is N > "+str(DLA_cut))
        plt.legend(loc=0)
        plt.xlim(self.minplot/100.,self.maxplot/100.)
        #Fits
        plt.tight_layout()
        plt.show()

    def plot_gas_vs_halo_mass(self):
        """Plot Gas mass vs total halo mass"""
        #Plot.
        self.ahalo.plot_gas_vs_halo_mass(label="Arepo",color=acol)
        self.ghalo.plot_gas_vs_halo_mass(label="Gadget",color=gcol)
        #Axes
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_rel_sigma_DLA(self,DLA_cut=20.3, min_sigma=15.):
        """
        Get the change in sigma_DLA for a particular halo.
        and the mass of each halo averaged across arepo and gadget.
        DLA_cut is the column density above which to consider a DLA
        min_sigma is the minimal sigma_DLA to look at (in grid cell units)
        """
        aDLA=self.ahalo.get_sigma_DLA(DLA_cut)
        gDLA=self.ghalo.get_sigma_DLA(DLA_cut)
        rDLA=np.empty(np.size(aDLA))
        rmass=np.empty(np.size(aDLA))
        cell_area=(2*self.ahalo.maxdist/self.ahalo.ngrid)**2
        for ii in xrange(0,np.size(aDLA)):
            gg=self.ghalo.identify_eq_halo(self.ahalo.sub_mass[ii],self.ahalo.sub_cofm[ii])
            if np.size(gg) > 0 and aDLA[ii]+gDLA[gg] > min_sigma*cell_area:
                rDLA[ii] = aDLA[ii]-gDLA[gg]
                rmass[ii]=0.5*(self.ahalo.sub_mass[ii]+self.ghalo.sub_mass[gg])
            else:
                rDLA[ii]=np.NaN
                rmass[ii]=np.NaN
        return (rmass,rDLA)


    def plot_rel_sigma_DLA(self):
        """Plot sigma_DLA against mass. Figure 10."""
#         (rmass,rDLA)=self.get_rel_sigma_DLA(17,25)
#         ind=np.where(np.isnan(rDLA) != True)
#         plt.semilogx(rmass[ind],rDLA[ind],'o',color="green",label="N_HI> 17")
        (rmass,rDLA)=self.get_rel_sigma_DLA(20.3,30.)
        ind=np.where(np.isnan(rDLA) != True)
        plt.semilogx(rmass[ind],rDLA[ind],'o',color="blue",label="N_HI> 20.3")
        #Axes
        plt.xlim(self.minplot,self.maxplot)
        plt.xlabel(r"Mass ($M_\odot$/h)")
        plt.ylabel(r"$\sigma_\mathrm{DLA}$ (Arepo) - $\sigma_\mathrm{DLA}$ (Gadget) (kpc$^2$/h$^2$)")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

#     def plot_dN_dla(self,Mmin=1e9,Mmax=1e13):
#         """Plots dN_DLA/dz for the halos. Figure 11"""
#         Mmax=np.min([Mmax,10**self.aDLAdz.log_mass_lim[1]])
#         mass=np.logspace(np.log10(Mmin),np.log10(Mmax),num=100)
#         aDLA_dz_tab = np.empty(np.size(mass))
#         gDLA_dz_tab = np.empty(np.size(mass))
#         for (i,m) in enumerate(mass):
#             aDLA_dz_tab[i] = self.aDLAdz.get_N_DLA_dz(m)
#             gDLA_dz_tab[i] = self.gDLAdz.get_N_DLA_dz(m)
#         print "AREPO: alpha=",self.aDLAdz.alpha," beta=",self.aDLAdz.beta
#         print "GADGET: alpha=",self.gDLAdz.alpha," beta=",self.gDLAdz.beta
#         plt.loglog(mass,aDLA_dz_tab,color=acol,label="Arepo",ls=astyle)
#         plt.loglog(mass,gDLA_dz_tab,color=gcol,label="Gadget",ls=gstyle)
#         plt.xlabel(r"Mass ($M_\odot$/h)")
#         plt.ylabel(r"$dN_{DLA} / dz (> M_\mathrm{tot})$")
#         plt.legend(loc=3)
#         plt.xlim(self.minplot,self.maxplot)
#         plt.tight_layout()
#         plt.show()

    def plot_column_density(self,minN=17,maxN=25.):
        """Plots the column density distribution function. Figures 12 and 13"""
        (aNHI,af_N)=self.ahalo.column_density_function(0.4,minN,maxN)
        (gNHI,gf_N)=self.ghalo.column_density_function(0.4,minN,maxN)
        plt.loglog(aNHI,af_N,color=acol, ls=astyle,label="Arepo")
        plt.loglog(gNHI,gf_N,color=gcol, ls=gstyle,label="Gadget")
        #Make the ticks be less-dense
        #ax=plt.gca()
        #ax.xaxis.set_ticks(np.power(10.,np.arange(int(minN),int(maxN),2)))
        #ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(af_N[-1])),int(np.log10(af_N[0])),2)))
        plt.xlabel(r"$N_{HI} (\mathrm{cm}^{-2})$")
        plt.ylabel(r"$f(N) (\mathrm{cm}^2)$")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

    def plot_radial_profile(self,minM=2e11,maxM=1e12,minR=0,maxR=100.):
        """Plots the radial density of neutral hydrogen for all halos stacked in the mass bin.
        """
        Rbins=np.linspace(minR,maxR,20)
        try:
            aRprof=[self.ahalo.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1]) for i in xrange(0,np.size(Rbins)-1)]
            gRprof=[self.ghalo.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1]) for i in xrange(0,np.size(Rbins)-1)]
            plt.semilogy(Rbins[0:-1],aRprof,color=acol, ls=astyle,label="Arepo HI")
            plt.semilogy(Rbins[0:-1],gRprof,color=gcol, ls=gstyle,label="Gadget HI")
            #If we didn't load the HI grid this time
        except AttributeError:
            pass
        #Gas profiles
        try:
            agRprof=[self.ahalo.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1],True) for i in xrange(0,np.size(Rbins)-1)]
            ggRprof=[self.ghalo.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1],True) for i in xrange(0,np.size(Rbins)-1)]
            plt.semilogy(Rbins[0:-1],agRprof,color="brown", ls=astyle,label="Arepo Gas")
            plt.semilogy(Rbins[0:-1],ggRprof,color="orange", ls=gstyle,label="Gadget Gas")
        except AttributeError:
            pass
        #Make the ticks be less-dense
        #ax=plt.gca()
        #ax.xaxis.set_ticks(np.power(10.,np.arange(int(minN),int(maxN),2)))
        #ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(af_N[-1])),int(np.log10(af_N[0])),2)))
        plt.xlabel(r"R (kpc/h)")
        plt.ylabel(r"Density $N_HI$ (cm$^{-1}$)")
        plt.legend(loc=1)
        plt.tight_layout()
        plt.show()

    def plot_rel_column_density(self,minN=17,maxN=25.):
        """Plots the column density distribution function. Figures 12 and 13"""
        (aNHI,af_N)=self.ahalo.column_density_function(0.4,minN,maxN)
        (gNHI,gf_N)=self.ghalo.column_density_function(0.4,minN,maxN)
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

#     def plot_halo_mass_func(self):
#         """Plots the halo mass function as well as Sheth-Torman. Figure 5."""
#         mass=np.logspace(np.log10(self.minplot),np.log10(self.maxplot),51)
#         shdndm=[self.aDLAdz.halo_mass.dndm(mm) for mm in mass]
#         adndm=np.empty(50)
#         gdndm=np.empty(50)
#         for ii in range(0,50):
#             adndm[ii]=self.ahalo.get_dndm(mass[ii],mass[ii+1])
#             gdndm[ii]=self.ghalo.get_dndm(mass[ii],mass[ii+1])
#         plt.loglog(mass,shdndm,color="black",ls='--',label="Sheth-Tormen")
#         plt.loglog(mass[0:-1],adndm,color=acol,ls=astyle,label="Arepo")
#         plt.loglog(mass[0:-1],gdndm,color=gcol,ls=gstyle,label="Gadget")
#         #Make the ticks be less-dense
#         ax=plt.gca()
#         ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(shdndm[-1])),int(np.log10(shdndm[0])),2)))
#
#         plt.ylabel(r"dn/dM (h$^4$ $M^{-1}_\odot$ Mpc$^{-3}$)")
#         plt.xlabel(r"Mass ($M_\odot$/h)")
#         plt.legend(loc=0)
#         plt.xlim(self.minplot,self.maxplot)
#         plt.tight_layout()
#         plt.show()
