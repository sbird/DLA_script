#!/usr/bin/env python
# vim: set fileencoding=UTF-8 :

"""
Plot P_F(k)
"""

import numpy
import scipy.interpolate
import math
import re
import glob
import matplotlib.pyplot as plt

def plot_flux_power(flux_power,box,zz,om,H0):
    """Plot the flux power spectrum in h/Mpc units"""
    #Units:
    #We need to divide by the box to get it into 1/Mpc units
    #and then multiply by the hubble parameter to be in 1/(km/s)
    scale=(1.0+zz)/(Hubble(zz,om,H0)*box/(H0/100.0))
    #Adjust Fourier convention.
    k=flux_power[1:,0]/box*2.0*math.pi
    PF=flux_power[1:,1]*box  #*((1+zz)/3.2)**3
    #plt.loglog(k, PF, linestyle="-")
#     xlim(k[0],k[-1])
    return (k,PF)

def Hubble(zz,om,H0):
    """Hubble function"""
    return H0*math.sqrt(om*(1+zz)**3+(1-om))

def MacDonaldPF(sdss, fbar,zz):
    """Plot the flux power spectrum from SDSS data, velocity units"""
    psdss=sdss[numpy.where(sdss[:,0] == zz)][:,1:3]
    PF=psdss[:,1]*fbar**2
    k=psdss[:,0]
    return (k, PF)


def PlotDiff(bigx,bigy, smallx,smally):
    """Plot the geometric difference between two flux powers"""
    inds=numpy.where(smallx >=bigx[0])
    smallx=smallx[inds]
    diff=smallx
    newstuff=scipy.interpolate.interpolate.interp1d(bigx,bigy)
    diff=smally[inds]/newstuff(smallx)
    return (smallx,diff)

#Mean flux is from the Kim et al paper 0711.1862
# (0.0023±0.0007)(1+z)^(3.65±0.21)
pfdir='/home/spb/scratch/ComparisonProject/'
#if len(sys.argv) > 1:
#    pfdir=sys.argv[1]
#else:
#    print "Usage: plot_flux_power.py flux_power_dir\n"
#    sys.exit(2)
tmp=numpy.loadtxt(pfdir+'redshifts.txt')
zzz=tmp[:,1]

def pfplots(num='100', ls="-"):
    """Plot a bundle of flux power spectra from Arepo and Gadget"""
    tdir=pfdir+'Gadget/snap_'+str(num).rjust(3,'0')+'_flux_power.txt'
    fluxpower=glob.glob(tdir)
    if (len(fluxpower) == 0):
        print "No flux power spectra found in "+tdir

    for pf in fluxpower:
        #Get header information
        z=zzz[int(num)]
        om=0.27
        box=20.
        H0=70
        #Plot the simulation output
        flux_power=numpy.loadtxt(pf)
        (simk, simPF)=plot_flux_power(flux_power,box,z,om,H0)
        arpf = re.sub("Gadget/","Arepo_no_rescale/",pf)
        flux_power=numpy.loadtxt(arpf)
        (arsimk, arsimPF)=plot_flux_power(flux_power,box,z,om,H0)
        plt.ylabel(r"$\mathrm{P}_\mathrm{F}(k) $ (Mpc/h)",size=22)
        #Obs. limit is 0.02 at present
        plt.loglog(simk,simPF,ls=ls,label='Gadget: z='+str(round(z,2)),color="blue",lw=8)
        plt.loglog(arsimk,arsimPF,ls=ls,label='Arepo: z='+str(round(z,2)),color="red",lw=4)

def pfrelplots(num='100',ls="-"):
    """Plot a bundle of flux power spectra from Arepo and Gadget"""
    tdir=pfdir+'Gadget/snap_'+str(num).rjust(3,'0')+'_flux_power.txt'
    fluxpower=glob.glob(tdir)
    if (len(fluxpower) == 0):
        print "No flux power spectra found in "+tdir

    for pf in fluxpower:
        #Get header information
        z=zzz[int(num)]
        om=0.27
        box=20.
        H0=70
        #Plot the simulation output
        flux_power=numpy.loadtxt(pf)
        (simk, simPF)=plot_flux_power(flux_power,box,z,om,H0)
        arpf = re.sub("Gadget/","Arepo_no_rescale/",pf)
        flux_power=numpy.loadtxt(arpf)
        (arsimk, arsimPF)=plot_flux_power(flux_power,box,z,om,H0)
#         plt.ylabel(r"$\delta P_F$ (%)")
        plt.xlabel(r"k (h/Mpc)",size=22)
        #Obs. limit is 0.02 at present
        plt.semilogx(simk,100*(simPF/arsimPF-1),label='z='+str(round(z,2)),ls=ls,color="black",lw=4)
#         plt.xlim(simk[0],0.03)

def pdfplots(num='100',ls="-"):
    """Plot a bundle of flux PDF's from Arepo and Gadget"""
    tdir=pfdir+'Gadget/snap_'+str(num).rjust(3,'0')+'_flux_pdf.txt'
    fluxpower=glob.glob(tdir)
    if (len(fluxpower) == 0):
        print "No flux pdf found in "+tdir
    z=str(round(zzz[int(num)],2))
    for pf in fluxpower:
        #Get header information
        #Plot the simulation output
        pdf=numpy.loadtxt(pf)

        arpf = re.sub("Gadget/","Arepo/",pf)
        pdf_ar=numpy.loadtxt(arpf)
#         ar256 = re.sub("Arepo/","Arepo_256/",pf)
#         pdf_ar256=numpy.loadtxt(ar256)
        plt.semilogy(pdf[:,0]/20., pdf[:,1],label='Gadget: z='+z,ls=ls,color="blue",lw=8)
        plt.semilogy(pdf_ar[:,0]/20., pdf_ar[:,1],label='Arepo: z='+z,ls=ls,color="red",lw=4)
#         plt.semilogy(pdf_ar256[:,0]/20., pdf_ar256[:,1],label='Arepo 256: z='+z,ls='..',color=color)
#       plt.ylabel(r"Flux PDF")
#     plt.xlim(0,1)
#     plt.xlabel("Flux")
#     plt.ylim(0.09,10)
#     plt.yticks((0.1,1,10),('0.1','1.0','10'))

def pdfrelplots(num='100',ls="-"):
    """Plot a bundle of flux PDF's from Arepo and Gadget"""
    tdir=pfdir+'Gadget/snap_'+str(num).rjust(3,'0')+'_flux_pdf.txt'
    fluxpower=glob.glob(tdir)
    if (len(fluxpower) == 0):
        print "No flux pdf found in "+tdir
    z=str(round(zzz[int(num)],2))
    for pf in fluxpower:
        #Get header information
        #Plot the simulation output
        pdf=numpy.loadtxt(pf)

        arpf = re.sub("Gadget/","Arepo/",pf)
        pdf_ar=numpy.loadtxt(arpf)
#         ar256 = re.sub("Gadget/","Arepo_256/",pf)
#         pdf_ar256=numpy.loadtxt(ar256)
#         gad256 = re.sub("Gadget/","Gadget_256/",pf)
#         pdf_gad256=numpy.loadtxt(gad256)
#         plt.plot(pdf_ar256[:,0]/20., pdf_ar256[:,1]/pdf_gad256[:,1],label='Ratio: z='+z,ls=':',color=color)
        plt.plot(pdf_ar[:,0]/20., 100*(pdf[:,1]/pdf_ar[:,1]-1),label='Ratio: z='+z,ls=ls,color="black",lw=4)
#     plt.ylabel(r"Rel Flux PDF")
#     plt.xlim(0,1)
#     plt.xlabel("Flux")
#     plt.ylim(0.95,1.05)
