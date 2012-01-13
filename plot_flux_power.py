#!/usr/bin/env python
# vim: set fileencoding=UTF-8 :

"""
Plot P_F(k)
"""

import numpy
import math
import re
import glob
from matplotlib.pyplot import *

def plot_flux_power(flux_power,box,zz,om,H0):
        bins=numpy.shape(flux_power)[0]
        #Units:
        #We need to divide by the box to get it into 1/Mpc units
        #and then multiply by the hubble parameter to be in 1/(km/s)
        scale=(1.0+zz)/(Hubble(zz,om,H0)*box/(H0/100.0))
        #Adjust Fourier convention.
        k=flux_power[1:,0]*scale*2.0*math.pi
        PF=flux_power[1:,1]/scale  #*((1+zz)/3.2)**3
        #loglog(k, PF, linestyle="-")
#         xlim(k[0],k[-1])
        return (k,PF)

def Hubble(zz,om,H0):
        return H0*math.sqrt(om*(1+zz)**3+(1-om))

def MacDonaldPF(sdss, fbar,zz):
        psdss=sdss[numpy.where(sdss[:,0] == zz)][:,1:3]
        PF=psdss[:,1]*fbar**2
        k=psdss[:,0]
        return (k, PF)


def PlotDiff(bigx,bigy, smallx,smally):
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
#        pfdir=sys.argv[1]
#else:
#        print "Usage: plot_flux_power.py flux_power_dir\n"
#        sys.exit(2)
tmp=np.loadtxt(pfdir+'redshifts.txt')
zz=tmp[:,1]

def pfplots(num='100'):
        tdir=pfdir+'Gadget/snap_'+str(num).rjust(3,'0')+'_flux_power.txt'
        fluxpower=glob.glob(tdir)
        if (len(fluxpower) == 0):
            print "No flux power spectra found in "+tdir
        
        for pf in fluxpower:
                #Get header information
                z=zz[int(num)]
                om=0.27
                box=20.
                H0=70
                #Plot the simulation output
                flux_power=numpy.loadtxt(pf)
                (simk, simPF)=plot_flux_power(flux_power,box,z,om,H0)
                arpf = re.sub("Gadget/","Arepo/",pf)
                flux_power=numpy.loadtxt(arpf)
                (arsimk, arsimPF)=plot_flux_power(flux_power,box,z,om,H0)
                #Plot the observational determination from MacDonald.
             #   fbar=math.exp(-0.0023*(1+zz)**3.65)
             #   (macdk, macdpf)=MacDonaldPF(sdss,fbar,zz)
             #   plot(macdk, macdpf, color="red")
             #   savefig(base+".pdf")
             #   clf()
                ylabel(r"$P_F(k) $")
                xlabel(r"$k (s/km)$")
                #Obs. limit is 0.02 at present
               # semilogx(simk[ind],simPF[ind]/arsimPF[ind],label='z='+str(round(z,2)))
                loglog(simk,simPF,ls='--',label='Gadget: z='+str(round(z,2)))
                loglog(simk,arsimPF,label='Arepo: z='+str(round(z,2)))
#                 xlim(simk[0],0.03)

def pdfplots(num='100'):
        tdir=pfdir+'Gadget/snap_'+str(num).rjust(3,'0')+'_flux_pdf.txt'
        fluxpower=glob.glob(tdir)
        if (len(fluxpower) == 0):
            print "No flux pdf found in "+tdir
        z=str(round(zz[int(num)],2)) 
        for pf in fluxpower:
                #Get header information
                #Plot the simulation output
                pdf=numpy.loadtxt(pf)

                arpf = re.sub("Gadget/","Arepo/",pf)
                pdf_ar=numpy.loadtxt(arpf)
                ylabel(r"Flux PDF")
                plot(pdf[:,0], pdf[:,1],label='Gadget: z='+z,ls='--')
                plot(pdf_ar[:,0], pdf_ar[:,1],label='Arepo: z='+z)
        xlim(0.5,20.5)
