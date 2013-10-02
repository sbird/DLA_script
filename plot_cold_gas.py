# -*- coding: utf-8 -*-
"""Plot the HI fraction as a function of density"""

import cold_gas
import math
import hsml
import numpy as np
import hdfsim
import matplotlib.pyplot as plt

class ColdGas:
    def __init__(self, num, base):
        """Plot various things with the cold gas fraction"""
        ff = hdfsim.get_file(num, base,0)
        self.redshift = ff["Header"].attrs["Redshift"]
        self.box = ff["Header"].attrs["BoxSize"]
        self.hubble = ff["Header"].attrs["HubbleParam"]
        ff.close()
        self.gas=cold_gas.RahmatiRT(self.redshift, self.hubble)
        #self.yaj=cold_gas.YajimaRT(self.redshift, self.hubble)
        self.num = num
        self.base = base

    def plot_neutral_frac(self,files=0):
        """Plot the neutral fraction from both code and annalytic approx"""
        ff = hdfsim.get_file(self.num, self.base,files)
        bar = ff["PartType0"]
        nH = self.gas.get_code_rhoH(bar)
        nH0_code = self.gas.code_neutral_fraction(bar)
        nH0_rah = self.gas.get_reproc_HI(bar)
        bin_edges = 10**np.arange(-6,3,0.5)

        (cen,nH0_code_bin) = self.binned_nH(bin_edges, nH,nH0_code)
        #(cen,nH0_yaj_bin) = self.binned_nH(bin_edges, nH,nH0_yaj)
        (cen,nH0_rah_bin) = self.binned_nH(bin_edges, nH,nH0_rah)
        ff.close()
        plt.loglog(cen, nH0_code_bin, color="red", label="Arepo")
        plt.loglog(cen, nH0_rah_bin,color="blue", label="Rahmati fit")
        #plt.loglog(cen, nH0_yaj_bin,color="green")

    def binned_nH(self,bin_edges, nH, nH0):
        """Find the median value for nH0 in some nH bins"""
        centers = np.zeros(np.size(bin_edges)-1)
        nH0_bin = np.zeros(np.size(bin_edges)-1)
        for i in xrange(0,np.size(bin_edges)-1):
            ind = np.where(np.logical_and(nH < bin_edges[i+1], nH > bin_edges[i]))
            nH0_bin[i] = np.median(nH0[ind])
            centers[i] = (bin_edges[i+1]-bin_edges[i])/2.
        return (centers, nH0_bin)

    def rho_crit(self):
        """Get the critical density at z=0 in units of g cm^-3"""
        #H in units of 1/s
        h100=3.2407789e-18*self.hubble
        #G in cm^3 g^-1 s^-2
        grav=6.672e-8
        rho_crit=3*h100**2/(8*math.pi*grav)
        return rho_crit

    def omega_gas(self):
        """Compute Omega_gas, the sum of the hydrogen mass, divided by the critical density.
        """
        mass = 0.
        kpchincm = 1./(1+self.redshift)
        self.protonmass=1.67262178e-24
        for files in np.arange(0,500):
            try:
                ff = hdfsim.get_file(self.num, self.base,files)
            except IOError:
                break
            bar = ff["PartType0"]
            nH = self.gas.get_code_rhoH(bar)

            pvol = 4./3.*math.pi*(hsml.get_smooth_length(bar)*kpchincm)**3
            #Total mass of H in g
            ind = np.where(nH < 0.1)
            mass += np.sum(nH[ind]*pvol[ind])*protonmass
            #mass += np.sum(nH*pvol)*protonmass
        #Total volume of the box in comoving cm^3
        volume = (self.box)**3
        #Total mass of HI * m_p / r_c
        omega_g=mass/volume/self.rho_crit()
        return omega_g

