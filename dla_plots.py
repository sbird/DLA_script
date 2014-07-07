# -*- coding: utf-8 -*-
"""
This is a module for making plots like those in Tescari & Viel, based on the data gathered in the Halohi module
Figures implemented:
    5,6,9,10-13
Possible but not implemented:
    14
        """

import halohi
import boxhi
import numpy as np
import os.path as path
import math
import bias
import matplotlib.pyplot as plt
from brokenpowerfit import powerfit
import matplotlib.colors
from cubehelix import cubehelix, cubehelix_r

import halo_mass_function

gcol="blue"
acol="red"
acol2="cyan"
gcol2="magenta"
rcol="black"
astyle="-"
gstyle="--"

#Modified jet that works better in B&W
jet2 =   {'red':   ((0., 1, 1),(0.2, 0, 0), (0.5, 1, 1), (0.75,1, 1),
                   (1, 0.5, 0.5)),
        'green': ((0., 1, 1),  (0.2,1, 1), (0.5,1, 1),
                  (0.75,0,0), (1, 0, 0)),
        'blue':  ((0., 1, 1), (0.2, 1, 1), (0.5,0, 0),
                  (1, 0, 0))}
spb_jet2 = matplotlib.colors.LinearSegmentedColormap('spb_jet2',jet2,256)

#These are parameters for the analytic fits for the DLA abundances.
#breakpoint is at 10^10.5
#For the CIC kernel
#arepo_halo_p = {
#         90  : [ 0.44788372118 , 33.9815447137 , 147.019872918 , 782.356835854 , 2.43699726448 , ],
#         91  : [ 0.44788372118 , 33.9815447137 , 147.019872918 , 782.356835854 , 2.43699726448 , ],
#         124  : [ 0.545131805553 , 33.8286589585 , 79.2409360956 , 7.3077275913 , 4.44331594943 , ],
#         141  : [ 0.545131805553 , 33.8286589585 , 79.2409360956 , 7.3077275913 , 4.44331594943 , ],
#         191  : [ 0.536604868665 , 33.880052174 , 60.3190730406 , -987.746517091 , 0.801277260999 , ],
#         }
#gadget_halo_p = {
#          90  : [ 0.400712532199 , 34.1198998574 , 90.8083441527 , 1172.67416847 , 3.28402808298 , ],
#          91  : [ 0.400712532199 , 34.1198998574 , 90.8083441527 , 1172.67416847 , 3.28402808298 , ],
#          124  : [ 0.584178906377 , 33.4164257889 , -43.6910928625 , 328.651366054 , 2.94190923713 , ],
#          141  : [ 0.584178906377 , 33.4164257889 , -43.6910928625 , 328.651366054 , 2.94190923713 , ],
#          191  : [ 0.799269775395 , 32.3381518149 , -76.8165819708 , 42.1669345832 , 2.55134740883 , ],
#          }
#For the SPH kernel
arepo_halo_p = {
         90  : [ 0.593228633746 , 33.1299984533 , 74.0493146085 , 1497.00861204 , 1.068627058 , ],
         141  : [ 0.495570003992 , 33.8061311491 , 101.01513697 , -474.362361738 , 0.528843128887 , ],
         191  : [ 0.518273575979 , 33.6289279888 , 66.7295698539 , -1124.56210394 , 0.786777504418 , ],
         }
gadget_halo_p = {
          90  : [ 0.429181628186 , 33.4591228946 , 79.8087750645 , 1073.24697702 , 2.97141901891 , ],
          141  : [ 0.6252104913 , 32.6599023717 , -27.2718078548 , 315.755545603 , 2.73050761493 , ],
          191  : [ 0.84906246444 , 31.4994150987 , -49.9975217441 , 20.1374668372 , 2.50768504269 , ],
          }

def tab_to_latex():
    """Convert above table to latex format"""
    i = 4
    for snap in (90, 141, 191):
        print str(i),"  & Arepo ",
        for jj in arepo_halo_p[snap]:
            print " & ",sig_fig(jj,2),
        print "\\\\ "
        print str(i),"  & Gadget ",
        for jj in gadget_halo_p[snap]:
            print " & ",sig_fig(jj,2),
        print "\\\\ "
        i-=1


def sig_fig(num,figs=3):
    """Round a number to figs significant figures"""
    #How many digits does number have?
    norm=np.floor(np.log10(np.abs(num)))
    rnded=np.round(num,int(figs-norm))
    if norm >= figs:
        return str(int(rnded))
    else:
        return str(rnded)

def pr_num(num,rnd=2):
    """Return a string rep of a number"""
    return str(np.round(num,rnd))


def tight_layout_wrapper():
    """Wrap tight_layout for matplotlib backends like ps which don't have it"""
    try:
        plt.tight_layout()
    except AttributeError:
        pass

class PrettyHalo(halohi.HaloHI):
    """
    Derived class with extra methods for plotting a pretty (high-resolution) picture of the grid around a halo.
    """

    def plot_pretty_something(self,num,grid,bar_label):
        """
        Plots a pretty (high-resolution) picture of the grid around a halo.
        Helper for the other functions.
        """
        #Plot a figure
        vmax=np.max([np.max(grid),25.5])
        maxdist = self.sub_radii[num]
        plt.imshow(grid,origin='lower',extent=(-maxdist,maxdist,-maxdist,maxdist),vmin=10,vmax=vmax,cmap=spb_jet2)
        bar=plt.colorbar()#use_gridspec=True)
        bar.set_label(bar_label)
        if (maxdist > 150) * (maxdist < 200):
            plt.xticks((-150,-75,0,75,150))
            plt.yticks((-150,-75,0,75,150))
        if maxdist > 300:
            plt.xticks((-300,-150,0,150,300))
            plt.yticks((-300,-150,0,150,300))
        plt.xlabel(r"y (kpc h$^{-1}$)")
        plt.ylabel(r"z (kpc h$^{-1}$)")
        tight_layout_wrapper()
        plt.show()

    def plot_pretty_halo(self,num=0):
        """
        Plots a pretty (high-resolution) picture of the grid around a halo.
        """
        self.plot_pretty_something(num,self.sub_nHI_grid[num],r"log$_{10}$ N$_\mathrm{HI}$ (cm$^{-2}$)")

    def plot_pretty_cut_halo(self,num=0,cut_LLS=17,cut_DLA=20.3):
        """
        Plots a pretty (high-resolution) picture of the grid around a halo.
        """
        cut_grid=np.array(self.sub_nHI_grid[num])
        ind=np.where(cut_grid < cut_LLS)
        cut_grid[ind]=10
        ind2=np.where((cut_grid < cut_DLA)*(cut_grid > cut_LLS))
        cut_grid[ind2]=12.
        ind3=np.where(cut_grid > cut_DLA)
        cut_grid[ind3]=20.3
        maxdist = self.sub_radii[num]
        plt.imshow(cut_grid,origin='lower',extent=(-maxdist,maxdist,-maxdist,maxdist),vmin=10,vmax=20.3, cmap=spb_jet2)
        if (maxdist > 150) * (maxdist < 200):
            plt.xticks((-150,-75,0,75,150))
            plt.yticks((-150,-75,0,75,150))
        if maxdist > 300:
            plt.xticks((-300,-150,0,150,300))
            plt.yticks((-300,-150,0,150,300))
        plt.xlabel(r"y (kpc h$^{-1}$)")
        plt.ylabel(r"z (kpc h$^{-1}$)")
        tight_layout_wrapper()
        plt.show()

    def plot_radial_profile(self,minM=3e11,maxM=1e12,minR=0,maxR=20.):
        """Plots the radial density of neutral hydrogen (and possibly gas) for a given halo,
        stacking several halo profiles together."""
        Rbins=np.linspace(minR,maxR,20)
        aRprof=[self.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1]) for i in xrange(0,np.size(Rbins)-1)]
        plt.plot(Rbins[0:-1],aRprof,color=acol, ls=astyle,label="HI")
        plt.xlabel(r"R (kpc h$^{-1}$)")
        plt.ylabel(r"Density $N_{HI}$ (kpc$^{-1}$)")
        tight_layout_wrapper()
        plt.show()

    def plot_column_density(self,minN=17,maxN=23.,color=acol, ls=astyle,moment=False):
        """Plots the column density distribution function. """
        (aNHI,af_N)=self.column_density_function(0.1,minN-1,maxN+1)
        if moment:
            paf_N = af_N*aNHI
        else:
            paf_N = af_N
        try:
            label = self.label
        except AttributeError:
            label=""
        plt.loglog(aNHI,paf_N,color=color, ls=ls, lw = 3, label=label)
        #Make the ticks be less-dense
        #ax=plt.gca()
        #ax.xaxis.set_ticks(np.power(10.,np.arange(int(minN),int(maxN),2)))
        #ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(af_N[-1])),int(np.log10(af_N[0])),2)))
        plt.xlabel(r"$N_\mathrm{HI} (\mathrm{cm}^{-2})$")
        if moment:
            plt.ylabel(r"$N_\mathrm{HI} f(N)$")
        else:
            plt.ylabel(r"$f(N)$")
#         plt.title(r"Column density function at $z="+pr_num(self.redshift,1)+"$")
        plt.xlim(10**minN, 10**maxN)
        plt.ylim(1e-27,1e-18)
        if moment:
            plt.ylim(5e-5,1)
#         plt.legend(loc=0)
#         tight_layout_wrapper()
        plt.show()

    def plot_column_density_breakdown(self,minN=17,maxN=23., color="black"):
        """Plots the column density distribution function, broken down into halos. """
        (aNHI,tot_af_N)=self.column_density_function(0.1,minN-1,maxN+1)
        (aNHI,af_N)=self.column_density_function(0.1,minN-1,maxN+1,minM=11, maxM=14)
        plt.loglog(aNHI,af_N/tot_af_N,color=color, ls="-",label="Big",lw=4)
        (aNHI,af_N)=self.column_density_function(0.1,minN-1,maxN+1,minM=10,maxM=11)
        plt.loglog(aNHI,af_N/tot_af_N,color=color, ls="--",label="Middle",lw=4)
        try:
            (aNHI,af_N)=self.column_density_function(0.1,minN-1,maxN+1,minM=9,maxM=10)
            plt.loglog(aNHI,af_N/tot_af_N,color=color, ls=":",label="Small",lw=4)
        except IndexError:
            pass
        ax=plt.gca()
        ax.set_xlabel(r"$N_\mathrm{HI} (\mathrm{cm}^{-2})$",size=25)
        ax.set_ylabel(r"$f_\mathrm{halo}(N) / f_\mathrm{tot} (N) $",size=25)
        plt.xlim(10**minN, 10**maxN)
        plt.ylim(1e-2,1)
        tight_layout_wrapper()
        plt.show()

    def plot_sigma_DLA_median(self, DLA_cut=20.3,DLA_upper_cut=42.):
        """Plot the median and scatter of sigma_DLA against mass."""
        mass=np.logspace(np.log10(np.min(self.sub_mass)),np.log10(np.max(self.sub_mass)),num=7)
        abin_mass = np.empty(np.size(mass)-1)
        abin_mass = halohi.calc_binned_median(mass,self.sub_mass, self.sub_mass)
        (amed,aloq,aupq)=self.get_sigma_DLA_binned(mass,DLA_cut,DLA_upper_cut)
        #To avoid zeros
        aloq-=1e-2
        #Plot median sigma DLA
        plt.errorbar(abin_mass, amed,yerr=[aloq,aupq],fmt='^',color=acol,ms=15,elinewidth=4)

    def plot_sigma_DLA(self, DLA_cut=20.3,DLA_upper_cut=42.):
        """Plot sigma_DLA vs mass"""
        self.plot_sigma_DLA_contour(DLA_cut,DLA_upper_cut)
        self.plot_sigma_DLA_median(DLA_cut,DLA_upper_cut)
        #Plot Axes stuff
        ax=plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
#       ax.tick_params(labelsize=30)
        ax.set_xlabel(r"Mass ($M_\odot$ h$^{-1}$)")
        plt.ylabel(r"$\sigma_\mathrm{DLA}$ (kpc$^2$)")
        if DLA_cut == 20.3:
            plt.title(r"DLA cross-section at $z="+pr_num(self.redshift,1)+"$")
        if DLA_cut == 17.:
            ax.set_ylabel(r"$\sigma_\mathrm{LLS}$ (kpc$^2$)")
            plt.title(r"LLS cross-section at $z="+pr_num(self.redshift,1)+"$")
        #plt.xlim(self.minplot,self.maxplot)
        if DLA_cut < 19:
            plt.ylim(ymin=10)
        else:
            plt.ylim(ymin=1,ymax=10**4)
        #Fits
        tight_layout_wrapper()
        plt.show()

    def plot_sigma_DLA_contour(self, DLA_cut=20.3,DLA_upper_cut=42.):
        """Plot sigma_DLA against mass."""
        asigDLA=self.get_sigma_DLA(DLA_cut,DLA_upper_cut)
        #Plot sigma DLA
        #As contour
        ind = np.where(asigDLA > 0)
        (hist,xedges, yedges)=np.histogram2d(np.log10(self.sub_mass[ind]),np.log10(asigDLA[ind]),bins=(30,30))
        xbins=np.array([(xedges[i+1]+xedges[i])/2 for i in xrange(0,np.size(xedges)-1)])
        ybins=np.array([(yedges[i+1]+yedges[i])/2 for i in xrange(0,np.size(yedges)-1)])
        plt.contourf(10**xbins,10**ybins,hist.T,[1,1000],colors=("#cd5c5c",acol2),alpha=0.4)



class PrettyBox(boxhi.BoxHI,PrettyHalo):
    """
    As above but for the whole box grid
    """
    def __init__(self,snap_dir,snapnum,nslice=1,reload_file=False,savefile=None, label=""):
        boxhi.BoxHI.__init__(self,snap_dir,snapnum,nslice, reload_file=reload_file,savefile=savefile)
        self.label = label

    def plot_sigma_DLA(self, minpart = 0, dist=2., color=acol, color2="#cd5c5c"):
        """Plot sigma_DLA"""
        #Load defaults from file
        self._get_sigma_DLA(minpart, dist)
        self.plot_sigma_DLA_median(color=color)
        self.plot_sigma_DLA_model(color=color)
        print "field dlas:",self.field_dla
        ind = np.where(np.logical_and(self.sigDLA > 0, self.real_sub_mass > 0))
        (hist,xedges, yedges)=np.histogram2d(np.log10(self.real_sub_mass[ind]),np.log10(self.sigDLA[ind]),bins=(30,30))
        xbins=np.array([(xedges[i+1]+xedges[i])/2 for i in xrange(0,np.size(xedges)-1)])
        ybins=np.array([(yedges[i+1]+yedges[i])/2 for i in xrange(0,np.size(yedges)-1)])
        plt.contourf(10**xbins,10**ybins,hist.T,[1,1000],colors=(color2,color),alpha=0.4)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r"Halo Mass ($M_\odot$)")
        plt.ylabel(r"$\sigma_\mathrm{DLA}$ (kpc/h)$^2$")

    def plot_sigma_LLS(self, minpart = 0, dist=2.):
        """Plot sigma_DLA"""
        #Load defaults from file
        self._get_sigma_LLS(minpart, dist)
        print "field dlas:",self.field_lls
        ind = np.where(self.sigLLS > 0)
        (hist,xedges, yedges)=np.histogram2d(np.log10(self.real_sub_mass[ind]),np.log10(self.sigLLS[ind]),bins=(30,30))
        xbins=np.array([(xedges[i+1]+xedges[i])/2 for i in xrange(0,np.size(xedges)-1)])
        ybins=np.array([(yedges[i+1]+yedges[i])/2 for i in xrange(0,np.size(yedges)-1)])
        plt.contourf(10**xbins,10**ybins,hist.T,[1,1000],colors=("#cd5c5c",acol2),alpha=0.4)
        plt.yscale('log')
        plt.xscale('log')

    def plot_halo_hist(self, Mmin=1e8, Mmax=8e12, nbins=20, color="blue",ls="-",dla = True, minpart = 0, dist=2., plot_error=False, errfac=1.):
        """Plot a histogram of the halo masses of DLA hosts. Each bin contains the fraction
           of DLA cells associated with halos in this mass bin"""
        if dla:
            self._get_sigma_DLA(minpart, dist)
            ind = np.where(self.sigDLA > 0)
            sigs = self.sigDLA[ind]/1e6
        else:
            self._get_sigma_LLS(minpart, dist)
            ind = np.where(self.sigLLS > 0)
            sigs = self.sigLLS[ind]/1e6
        massbins = np.logspace(np.log10(Mmin), np.log10(Mmax), nbins+1)
        #Now we have a cross-section, we know how many DLA cells are associated with each halo.
        (hist,xedges)=np.histogram(np.log10(self.real_sub_mass[ind]),weights = sigs,bins=np.log10(massbins),density=False)
        if dla:
            bb = bias.HaloBias(self.redshift, self.omegam, 0.045, self.omegal, self.hubble, 0.97)
            ii = np.where(np.logical_and(self.real_sub_mass[ind] > 1e9, self.real_sub_mass[ind] < 1e12))
            biases = bb.halo_bias(self.real_sub_mass[ind][ii])
            dla_bias = np.sqrt(np.sum(sigs[ii]*biases**2)/ np.sum(sigs[ii]))
            print "DLA bias is:", dla_bias
        #For error bars
        xbins=np.array([(10**xedges[i+1]+10**xedges[i])/2 for i in xrange(0,np.size(xedges)-1)])
        nzind = np.where(hist > 0)
        plt.semilogx(xbins[nzind],hist[nzind], color=color, ls=ls, label=self.label)
        if plot_error:
            (nn,_)=np.histogram(np.log10(self.real_sub_mass[ind]),bins=np.log10(massbins))
            plt.semilogx(xbins[nzind],hist[nzind]*(1+1./np.sqrt(errfac*nn[nzind])), color="grey", ls=ls)
            plt.semilogx(xbins[nzind],hist[nzind]*(1-1./np.sqrt(errfac*nn[nzind])), color="grey", ls=ls)
        plt.xlabel(r"Halo Mass ($M_\odot$)")
        plt.ylabel(r"$\sigma_\mathrm{T}$ (Mpc/h)$^2$")

    def plot_halo_mass_func(self):
        """Plots the halo mass function from simulation as well as Sheth-Torman"""
        self.load_sigDLA()
        mass=np.logspace(8,13,51)
        halo_mass=halo_mass_function.HaloMassFunction(self.redshift,omega_m=self.omegam, omega_l=self.omegal, hubble=self.hubble,log_mass_lim=(7,15))
        shdndm=[halo_mass.dndm(mm) for mm in mass]
        adndm=np.array([self.get_dndm(mass[ii],mass[ii+1]) for ii in range(0,50)])
        plt.loglog(mass,shdndm,color="black",ls='--',label="Sheth-Tormen")
        plt.loglog(mass[0:-1],adndm,color=acol,ls=astyle,label="Arepo")
        #Make the ticks be less-dense
        ax=plt.gca()
        ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(shdndm[-1])),int(np.log10(shdndm[0])),2)))
        plt.ylabel(r"dn/dM (h$^4$ $M^{-1}_\odot$ Mpc$^{-3}$)")
        plt.xlabel(r"Mass ($M_\odot$ h$^{-1}$)")
        plt.legend(loc=0)
#         plt.xlim(self.minplot,self.maxplot)
        tight_layout_wrapper()
        plt.show()

    def plot_sigma_DLA_model(self,color="red"):
        """Plot my analytic model for the DLAs"""
        ind = np.where(self.real_sub_mass > 0)
        mass=np.logspace(8.5,12,num=100)
        #Plot Analytic Fit
        ap=self.get_sDLA_fit()
        mdiff=np.log10(mass)-ap[0]
        fit=(ap[2]*mdiff+ap[1])
#         asfit=broken_fit(ap, np.log10(mass))
        print "Fit: ",ap
        plt.loglog(mass,10**fit,color=color,ls="-",lw=3)

    def plot_sigma_DLA_median(self, DLA_cut=20.3,DLA_upper_cut=42.,color=acol):
        """Plot the median and scatter of sigma_DLA against mass."""
        ind = np.where(self.real_sub_mass > 0)
        mass=np.logspace(np.log10(np.min(self.real_sub_mass[ind])),np.log10(np.max(self.real_sub_mass[ind])),num=7)
        abin_mass = np.empty(np.size(mass)-1)
        abin_mass = halohi.calc_binned_median(mass,self.real_sub_mass[ind], self.real_sub_mass[ind])
        (amed,aloq,aupq)=self.get_sigma_DLA_binned(mass,DLA_cut,DLA_upper_cut)
        #To avoid zeros
        aloq-=1e-2
        #Plot median sigma DLA
        plt.errorbar(abin_mass, amed,yerr=[aloq,aupq],fmt='^',color=color,ms=15,elinewidth=4)

    def plot_dla_metallicity(self, nbins=40,color="blue", ls="-"):
        """Plot the distribution of DLA metallicities"""
        met = self.get_dla_metallicity()
        return self._plot_metallicity(met,nbins=nbins,color=color,ls=ls)

    def plot_lls_metallicity(self, nbins=40,color="blue",ls="-"):
        """Plot the distribution of LLS metallicities"""
        met = self.get_lls_metallicity()
        return self._plot_metallicity(met, nbins=nbins,color=color,ls=ls)

    def _plot_metallicity(self, met, nbins,color,ls,upper=0,lower=-3):
        """Plot the distribution of metallicities above"""
        bins=np.logspace(lower,upper,nbins)
        mbin = np.array([(bins[i]+bins[i+1])/2. for i in range(0,np.size(bins)-1)])
        #Abs. distance for entire spectrum
        hist = np.histogram(met,np.log10(bins),density=True)[0]
        plt.plot(np.log10(mbin),hist,color=color,ls=ls, label=self.label)
        plt.xlabel(r"log $(Z / Z_\odot)$")
        return (mbin,hist)

    def plot_species_fraction(self, species, ion, dla=True, nbins=40, color="blue", ls="-"):
        """Plot [X/HI] for X given by species and ion in DLAs or LLS"""
        met = self.get_ion_metallicity(species, ion, dla)
        return self._plot_metallicity(met, nbins=nbins,color=color,ls=ls)

    def plot_ion_corr(self, species, ion, dla=True,nbins=80,color="blue",ls="-",upper=1,lower=-1):
        """Plot the difference between the single-species ionisation and the metallicity from GFM_Metallicity"""
        if dla:
            met = self.get_dla_metallicity()
        else:
            met = self.get_lls_metallicity()
        ion_met = self.get_ion_metallicity(species, ion, dla)
        print 10**np.max(ion_met-met), 10**np.min(ion_met-met), 10**np.median(ion_met-met)
        return self._plot_metallicity(ion_met-met, nbins=nbins,color=color,ls=ls,upper=upper,lower=lower)

    def plot_impact_param(self, minM = 1e6, maxM=1e20, nbins=40,color="blue",ls="-"):
        """Plot the distribution of metallicities above"""
        impact = self.get_dla_impact_parameter(minM,maxM)
        bins=np.linspace(0,2,nbins)
        mbin = np.array([(bins[i]+bins[i+1])/2. for i in range(0,np.size(bins)-1)])
        #Abs. distance for entire spectrum
        hist = np.histogram(impact,bins,density=True)[0]
        plt.plot(mbin,hist,color=color,ls=ls, label=self.label)
        plt.xlabel(r"distance to halo ($R_\mathrm{vir}$)")
        return (mbin,hist)

    def plot_dla_mass_metallicity(self, color="blue"):
        """Plot host halo mass vs metallicity for DLAs"""
        (halo_mass, _, _) = self._load_halo(0)
        self._get_sigma_DLA(0,2)
        ind = np.where(self.dla_halo >= 0)
        masses = halo_mass[self.dla_halo[ind]]
        met = self.get_dla_metallicity()[ind]
        ind2 = np.where((met > -2.7)*(met < 0.5)*(masses > 10**9))
        met = met[ind2]
        masses = np.log10(masses[ind2])
        (H, xedges, yedges) = np.histogram2d(masses, met,bins=10,normed=True)
        xbins=np.array([(xedges[i+1]+xedges[i])/2 for i in xrange(0,np.size(xedges)-1)])
        ybins=np.array([(yedges[i+1]+yedges[i])/2 for i in xrange(0,np.size(yedges)-1)])
        plt.contourf(10**xbins,ybins,H.T,[0.1,1],colors=(color,"black"),alpha=0.5)
        #Get fit
        amed=halohi.calc_binned_median(xedges, masses, met)
        aupq=halohi.calc_binned_percentile(xedges, masses, met,75)-amed
        #Addition to avoid zeros
        aloq=amed - halohi.calc_binned_percentile(xedges, masses, met,100-75)
        err = (aupq+aloq)/2.
        #Arbitrary large values if err is zero
        ap = powerfit(xbins, amed, np.log10(err), breakpoint=10)
        mdiff=xbins-ap[0]
        fit=(ap[2]*mdiff+ap[1])
#         asfit=broken_fit(ap, np.log10(mass))
        print "Fit: ",ap
        #Plot median sigma DLA
        plt.errorbar(10**xbins, amed,yerr=[aloq,aupq],fmt='^',color=color,ms=15,elinewidth=4)
        plt.semilogx(10**xbins, fit,color=color,ls="-",lw=3)
        plt.ylabel(r"log $(Z / Z_\odot)$")
        plt.xlabel(r"Halo mass ($M_\odot$)")
        return (masses,met)

    def plot_colden_mass_breakdown(self,ncdbins=30):
        """Find the proportion of DLAs in each column density bin with various halo masses"""
        (halo_mass, _, _) = self._load_halo(0)
        self._get_sigma_DLA(0,2)
        ind = np.where(self.dla_halo >= 0)
        find = np.where(self.dla_halo < 0)
        masses = halo_mass[self.dla_halo[ind]]
        dlaval = self._load_dla_val(True)
        cdbins = np.linspace(20.3,np.max(dlaval),ncdbins+1)
        massbins = 10**np.arange(9,14)
        massbins[0] = 10**8
        nmassbins = np.size(massbins)-1
#         massbins = np.logspace(np.log10(np.min(masses))-0.1,np.log10(np.max(masses))+0.1,nmassbins+1)
        fractions = np.zeros([nmassbins+2,ncdbins])
        for cc in xrange(ncdbins):
            cind = np.where((dlaval[ind] > cdbins[cc])*(dlaval[ind] <= cdbins[cc+1]))
            for mm in xrange(nmassbins):
                mind = np.where((masses[cind] > massbins[mm])*(masses[cind] <= massbins[mm+1]))
                fractions[mm+1, cc] = np.max(np.shape(mind))
            #Field DLAs
            cind = np.where((dlaval[find] > cdbins[cc])*(dlaval[find] <= cdbins[cc+1]))
            fractions[nmassbins+1,cc] = np.max(np.shape(cind))

        cumfrac = np.cumsum(fractions,0)
        fractions/=cumfrac[-1,:]
        cumfrac/=cumfrac[-1,:]
        width = np.array([(-cdbins[i]+cdbins[i+1]) for i in range(0,np.size(cdbins)-1)])
        colors = ["red", "green", "blue", "black", "orange", "purple"]
        for mm in xrange(nmassbins):
            plt.bar(cdbins[:-1],fractions[mm+1,:],bottom=cumfrac[mm,:],width=width, color=colors[mm],label=pr_num(np.log10(massbins[mm]))+" - "+pr_num(np.log10(massbins[mm+1])), linewidth=0)
        plt.bar(cdbins[:-1],fractions[nmassbins+1,:],bottom=cumfrac[nmassbins,:],width=width, label="Field",color=colors[-1],linewidth=0)
        return (cdbins, massbins, fractions)

import halomet

class PrettyMetal(halomet.HaloMet,PrettyHalo):
    """
    As above but for the whole box grid
    """
    def __init__(self,snap_dir,snapnum,elem, ion,reload_file=False,savefile=None):
        halomet.HaloMet.__init__(self,snap_dir,snapnum,elem, ion, reload_file=reload_file,savefile=savefile)

    def plot_pretty_halo(self,num=0):
        """
        Plots a pretty (high-resolution) picture of the grid around a halo.
        Helper for the other functions.
        """
        #Plot a figure

        maxdist = self.sub_radii[num]
        plt.imshow(self.sub_nHI_grid[num],origin='lower',extent=(-maxdist,maxdist,-maxdist,maxdist),vmin=13,cmap=cubehelix_r)
        bar=plt.colorbar()#use_gridspec=True)
        bar.set_label(r"log$_{10}$ N$_\mathrm{Si}$ (amu/cm$^{-2}$)")
        if maxdist > 300:
            plt.xticks((-300,-150,0,150,300))
            plt.yticks((-300,-150,0,150,300))
        plt.xlabel(r"y (kpc h$^{-1}$)")
        plt.ylabel(r"z (kpc h$^{-1}$)")
        tight_layout_wrapper()
        plt.show()


class HaloHIPlots:
    """
    This class contains functions for plotting all the plots in
    Tescari and Viel which are derived from the grid of HI density around the halos.
    These are figs 10-13
    """
    def __init__(self,base,snapnum,minpart=400,minplot=1e9, maxplot=2e12,reload_file=False):
        #Get paths
        self.gdir=path.join(base,"Gadget")
        self.adir=path.join(base,"Arepo_ENERGY")
        #Get data
        self.ahalo=PrettyHalo(self.adir,snapnum,minpart,reload_file=reload_file)
#         self.ahalo.save_file()
        self.ghalo=PrettyHalo(self.gdir,snapnum,minpart,reload_file=reload_file)
#         self.ghalo.save_file()
        self.minplot=minplot
        self.maxplot=maxplot

    def plot_sigma_DLA_model(self,DLA_cut=20.3,DLA_upper_cut=42.):
        """Plot my analytic model for the DLAs"""
        mass=np.logspace(np.log10(np.min(self.ahalo.sub_mass)),np.log10(np.max(self.ahalo.sub_mass)),num=100)
        #Plot Analytic Fit
#         ap=self.ahalo.get_sDLA_fit()
#         gp=self.ghalo.get_sDLA_fit()
        ap=arepo_halo_p[self.ahalo.snapnum]
        gp=gadget_halo_p[self.ghalo.snapnum]
        asfit=self.ahalo.sDLA_analytic(mass,ap,DLA_cut)-self.ahalo.sDLA_analytic(mass,ap,DLA_upper_cut)
        gsfit=self.ghalo.sDLA_analytic(mass,gp,DLA_cut)-self.ghalo.sDLA_analytic(mass,gp,DLA_upper_cut)
#         print "Arepo: ",ap
#         print "Gadget: ",gp
        plt.loglog(mass,gsfit,color=gcol,ls=gstyle,lw=3)
        plt.loglog(mass,asfit,color=acol,ls=astyle,lw=3)

    def plot_sigma_DLA_median(self, DLA_cut=20.3,DLA_upper_cut=42.):
        """Plot the median and scatter of sigma_DLA against mass."""
        self.ahalo.plot_sigma_DLA_median(DLA_cut, DLA_upper_cut)
        self.ghalo.plot_sigma_DLA_median(DLA_cut, DLA_upper_cut)

    def plot_sigma_DLA(self, DLA_cut=20.3,DLA_upper_cut=42.):
        """Plot sigma_DLA vs mass"""
        self.plot_sigma_DLA_contour(DLA_cut,DLA_upper_cut)
        self.plot_sigma_DLA_model(DLA_cut,DLA_upper_cut)
        self.plot_sigma_DLA_median(DLA_cut,DLA_upper_cut)
        #Plot Axes stuff
        ax=plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
#       ax.tick_params(labelsize=30)
        ax.set_xlabel(r"Mass ($M_\odot$ h$^{-1}$)",size=25)
        plt.ylabel(r"$\sigma_\mathrm{DLA}$ (kpc$^2$)",size=25)
        if DLA_cut == 20.3:
            plt.title(r"DLA cross-section at $z="+pr_num(self.ahalo.redshift,1)+"$")
        if DLA_cut == 17.:
            ax.set_ylabel(r"$\sigma_\mathrm{LLS}$ (kpc$^2$)",size=25)
            plt.title(r"LLS cross-section at $z="+pr_num(self.ahalo.redshift,1)+"$")
        plt.xlim(self.minplot,self.maxplot)
        if DLA_cut < 19:
            plt.ylim(ymin=10)
        else:
            plt.ylim(ymin=1,ymax=10**4)
        #Fits
        tight_layout_wrapper()
        plt.show()

    def plot_sigma_DLA_contour(self, DLA_cut=20.3,DLA_upper_cut=42.):
        """Plot sigma_DLA against mass."""
        self.ahalo.plot_sigma_DLA_contour(DLA_cut, DLA_upper_cut)
        self.ghalo.plot_sigma_DLA_contour(DLA_cut, DLA_upper_cut)

    def get_rel_sigma_DLA(self,DLA_cut=20.3, DLA_upper_cut=42.,min_sigma=15.):
        """
        Get the change in sigma_DLA for a particular halo.
        and the mass of each halo averaged across arepo and gadget.
        DLA_cut is the column density above which to consider a DLA
        min_sigma is the minimal sigma_DLA to look at (in grid cell units)
        """
        aDLA=self.ahalo.get_sigma_DLA(DLA_cut,DLA_upper_cut)
        gDLA=self.ghalo.get_sigma_DLA(DLA_cut,DLA_upper_cut)
        rDLA=np.empty(np.size(aDLA))
        rmass=np.empty(np.size(aDLA))
        cell_area=(2*self.ahalo.sub_radii[0]/self.ahalo.ngrid[0])**2
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
        plt.xlabel(r"Mass ($M_\odot$ h$^{-1}$)")
        plt.ylabel(r"$\sigma_\mathrm{DLA}$ (Arepo) - $\sigma_\mathrm{DLA}$ (Gadget) (kpc$^2$ h$^{-2}$)")
        plt.legend(loc=0)
        tight_layout_wrapper()
        plt.show()

    def plot_dN_dla(self,Mmin=1e9,Mmax=1e13):
        """Plots dN_DLA/dz for the halos. Figure 11"""
        mass=np.logspace(np.log10(Mmin),np.log10(Mmax),num=100)
        aDLA_dz_tab = np.empty(np.size(mass))
        gDLA_dz_tab = np.empty(np.size(mass))
        for (i,m) in enumerate(mass):
            aDLA_dz_tab[i] = self.ahalo.get_N_DLA_dz(arepo_halo_p[self.ahalo.snapnum],m)
            gDLA_dz_tab[i] = self.ghalo.get_N_DLA_dz(gadget_halo_p[self.ghalo.snapnum],m)
        plt.loglog(mass,aDLA_dz_tab,color=acol,label="Arepo",ls=astyle,lw=4)
        plt.loglog(mass,gDLA_dz_tab,color=gcol,label="Gadget",ls=gstyle,lw=4)
        ax=plt.gca()
        ax.fill_between(mass, 10**(-0.7), 10**(-0.5),color='yellow')
        ax.set_xlabel(r"Mass ($M_\odot$ h$^{-1}$)",size=25)
        ax.set_ylabel(r"$\mathrm{dN}_\mathrm{DLA} / \mathrm{dz} (> M_\mathrm{tot})$",size=25)
#         plt.legend(loc=3)
        plt.xlim(Mmin,1e12)
        plt.ylim(10**(-2),1)
        tight_layout_wrapper()
        plt.show()
#         print "Arepo mean halo mass: ",self.ahalo.get_mean_halo_mass(arepo_halo_p[self.ahalo.snapnum])/1e10
#         print "Gadget mean halo mass: ",self.ghalo.get_mean_halo_mass(gadget_halo_p[self.ghalo.snapnum])/1e10

    def plot_column_density(self,minN=17,maxN=23.):
        """Plots the column density distribution function. Figures 12 and 13"""
        (aNHI,af_N)=self.ahalo.column_density_function(0.1,minN-1,maxN+1)
        (gNHI,gf_N)=self.ghalo.column_density_function(0.1,minN-1,maxN+1)
        plt.loglog(aNHI,af_N,color=acol, ls=astyle,label="Arepo",lw=6)
        plt.loglog(gNHI,gf_N,color=gcol, ls=gstyle,label="Gadget",lw=6)
#         (aNH2,af_NH2)=self.ahalo.column_density_function(0.4,minN-1,maxN+1,grids=1)
#         (gNH2,gf_NH2)=self.ghalo.column_density_function(0.4,minN-1,maxN+1,grids=1)
#         plt.loglog(aNH2,af_NH2,color=acol2, ls=astyle,label="Arepo")
#         plt.loglog(gNH2,gf_NH2,color=gcol2, ls=gstyle,label="Gadget")
        #Make the ticks be less-dense
        ax=plt.gca()
        #ax.xaxis.set_ticks(np.power(10.,np.arange(int(minN),int(maxN),2)))
        #ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(af_N[-1])),int(np.log10(af_N[0])),2)))
        ax.set_xlabel(r"$N_\mathrm{HI} (\mathrm{cm}^{-2})$",size=25)
        ax.set_ylabel(r"$f(N) (\mathrm{cm}^2)$",size=25)
        plt.title(r"Column density function at $z="+pr_num(self.ahalo.redshift,1)+"$")
        plt.xlim(10**minN, 10**maxN)
        plt.ylim(1e-26,1e-18)
#         plt.legend(loc=0)
        tight_layout_wrapper()
        plt.show()

    def plot_column_density_breakdown(self,minN=17,maxN=23.):
        """Plots the column density distribution function, broken down into halos. """
        (aNHI,tot_af_N)=self.ahalo.column_density_function(0.4,minN-1,maxN+1)
        (gNHI,tot_gf_N)=self.ghalo.column_density_function(0.4,minN-1,maxN+1)
        (aNHI,af_N)=self.ahalo.column_density_function(0.4,minN-1,maxN+1,minM=11)
        (gNHI,gf_N)=self.ghalo.column_density_function(0.4,minN-1,maxN+1,minM=11)
        plt.loglog(aNHI,tot_af_N/tot_gf_N,color="black", ls="-",label="Arepo",lw=4)
        plt.loglog(gNHI,gf_N/tot_gf_N,color=gcol, ls="-",label="Gadget",lw=7)
        plt.loglog(aNHI,af_N/tot_gf_N,color=acol, ls="-",label="Arepo",lw=4)
        (aNHI,af_N)=self.ahalo.column_density_function(0.4,minN-1,maxN+1,minM=10,maxM=11)
        (gNHI,gf_N)=self.ghalo.column_density_function(0.4,minN-1,maxN+1,minM=10,maxM=11)
        plt.loglog(gNHI,gf_N/tot_gf_N,color=gcol, ls="--",label="Gadget",lw=7)
        plt.loglog(aNHI,af_N/tot_gf_N,color=acol, ls="--",label="Arepo",lw=4)
        try:
            (aNHI,af_N)=self.ahalo.column_density_function(0.4,minN-1,maxN+1,minM=9,maxM=10)
            (gNHI,gf_N)=self.ghalo.column_density_function(0.4,minN-1,maxN+1,minM=9,maxM=10)
            plt.loglog(gNHI,gf_N/tot_gf_N,color=gcol, ls=":",label="Gadget",lw=7)
            plt.loglog(aNHI,af_N/tot_gf_N,color=acol, ls=":",label="Arepo",lw=4)
        except IndexError:
            pass
        #Make the ticks be less-dense
        ax=plt.gca()
        #ax.xaxis.set_ticks(np.power(10.,np.arange(int(minN),int(maxN),2)))
        #ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(af_N[-1])),int(np.log10(af_N[0])),2)))
        ax.set_xlabel(r"$N_\mathrm{HI} (\mathrm{cm}^{-2})$",size=25)
        ax.set_ylabel(r"$f_\mathrm{halo}(N) / f_\mathrm{GADGET} (N) $",size=25)
#         plt.title(r"Halo contribution to $f(N)$ at $z="+pr_num(self.ahalo.redshift,1)+"$")
        plt.xlim(10**minN, 10**maxN)
        plt.ylim(1e-2,2)
#         plt.legend(loc=0)
        tight_layout_wrapper()
        plt.show()

    def plot_radial_profile(self,minM=4e11,maxM=1e12,minR=0,maxR=40.):
        """Plots the radial density of neutral hydrogen for all halos stacked in the mass bin.
        """
        #Use sufficiently large bins
        scale = 10**43
        space=2.*self.ahalo.sub_radii[0]/self.ahalo.ngrid[0]
        if maxR/30. > space:
            Rbins=np.linspace(minR,maxR,20)
        else:
            Rbins=np.concatenate((np.array([minR,]),np.linspace(minR+np.ceil(2.5*space),maxR+space,maxR/np.ceil(space))))
        Rbinc = [(Rbins[i+1]+Rbins[i])/2 for i in xrange(0,np.size(Rbins)-1)]
        Rbinc=np.array([minR,]+Rbinc)
        try:
            ind = np.where(np.logical_and(self.ahalo.sub_mass > minM, self.ahalo.sub_mass < maxM))
            gind = np.where(np.logical_and(self.ghalo.sub_mass > minM, self.ghalo.sub_mass < maxM))
            print "No. of halos for ",minM," < M < ",maxM," Arepo: ",np.size(ind)," Gadget: ",np.size(gind)
            aRprof=[self.ahalo.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1])/scale for i in xrange(0,np.size(Rbins)-1)]
            gRprof=[self.ghalo.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1])/scale for i in xrange(0,np.size(Rbins)-1)]
            plt.semilogy(Rbinc,[aRprof[0],]+aRprof,color=acol, ls=astyle,label="Arepo HI",lw=4)
            plt.semilogy(Rbinc,[gRprof[0],]+gRprof,color=gcol, ls=gstyle,label="Gadget HI",lw=4)
            RR = np.linspace(minR,maxR,100)
            plt.semilogy(RR,1e-5+2*math.pi*RR/(1+self.ahalo.redshift)*self.ahalo.UnitLength_in_cm*10**20.3/scale,color="black", ls="-.",label="DLA density",lw=4)
            maxx=np.max((aRprof[0],gRprof[0]))
            #If we didn't load the HI grid this time
        except AttributeError:
            pass
#         #Gas profiles
#         try:
#             agRprof=[self.ahalo.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1],True) for i in xrange(0,np.size(Rbins)-1)]
#             ggRprof=[self.ghalo.get_stacked_radial_profile(minM,maxM,Rbins[i],Rbins[i+1],True) for i in xrange(0,np.size(Rbins)-1)]
#             plt.plot(Rbinc,[agRprof[0],]+agRprof,color="brown", ls=astyle,label="Arepo Gas")
#             plt.plot(Rbinc,[ggRprof[0],]+ggRprof,color="orange", ls=gstyle,label="Gadget Gas")
#             maxx=np.max((agRprof[0],ggRprof[0]))
#         except AttributeError:
#             pass
        #Make the ticks be less-dense
        #ax=plt.gca()
        #ax.xaxis.set_ticks(np.power(10.,np.arange(int(minN),int(maxN),2)))
        #ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(af_N[-1])),int(np.log10(af_N[0])),2)))
        plt.xlabel(r"R (kpc h$^{-1}$)")
        plt.ylabel(r"Radial Density ($10^{43}$ cm$^{-1}$)")
        #Crop the frame so we see the DLA cross-over point
        DLAdens=2*math.pi*Rbins[-1]*self.ahalo.UnitLength_in_cm*10**20.3
        if maxx > 20*DLAdens:
            plt.ylim(1e-2,20*DLAdens)
        else:
            plt.ylim(1e-2,5*np.floor(gRprof[0]/5)+5)
        plt.xlim(minR,maxR)
        tight_layout_wrapper()
        plt.show()

    def plot_rel_column_density(self,minN=17,maxN=23.):
        """Plots the column density distribution function. Figures 12 and 13"""
        (aNHI,af_N)=self.ahalo.column_density_function(0.4,minN-1,maxN+1)
        (_,gf_N)=self.ghalo.column_density_function(0.4,minN-1,maxN+1)
        plt.semilogx(aNHI,af_N/gf_N,label="Arepo / Gadget",color=rcol)
        #Make the ticks be less-dense
#         ax=plt.gca()
#         ax.xaxis.set_ticks(np.power(10.,np.arange(int(minN),int(maxN),3)))
        #ax.yaxis.set_ticks(np.power(10.,np.arange(int(np.log10(af_N[-1])),int(np.log10(af_N[0])),2)))
        plt.xlabel(r"$N_\mathrm{HI} (\mathrm{cm}^{-2})$")
        plt.ylabel(r"$ \delta f(N)$")
        plt.xlim(10**minN, 10**maxN)
#         plt.legend(loc=0)
        tight_layout_wrapper()
        plt.show()

    def plot_halo_mass_func(self):
        """Plots the halo mass function as well as Sheth-Torman. Figure 5."""
        mass=np.logspace(np.log10(self.minplot),np.log10(self.maxplot),51)
        shdndm=[self.ahalo.halo_mass.dndm(mm) for mm in mass]
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
        plt.xlabel(r"Mass ($M_\odot$ h$^{-1}$)")
        plt.legend(loc=0)
        plt.xlim(self.minplot,self.maxplot)
        tight_layout_wrapper()
        plt.show()

    def print_halo_fits(self):
        """Prints the fitted parameters for the sDLA model"""
        ap=self.ahalo.get_sDLA_fit()
        gp=self.ghalo.get_sDLA_fit()
        print "Arepo: "
        print self.ahalo.snapnum," : ",ap,","
        print "Gadget: "
        print self.ghalo.snapnum," : ",gp,","




