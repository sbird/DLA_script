"""Plot the HI fraction as a function of density"""

import cold_gas
import numpy as np
import hdfsim
import matplotlib.pyplot as plt

class ColdGas:
    def __init__(self, num, base):
        """Plot various things with the cold gas fraction"""
        ff = hdfsim.get_file(num, base,0)
        self.redshift = ff["Header"].attrs["Redshift"]
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
        nH0_rah = self.gas.get_reproc_rhoHI(bar)/nH
        #nH0_yaj = self.yaj.get_reproc_rhoHI(bar)/nH
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

