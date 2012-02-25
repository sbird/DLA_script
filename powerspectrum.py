"""Module for finding the power spectrum of an array, evenly binned in k"""

import numpy as np


def KVAL(n, d):
    """Convert fourier order into real coordinate"""
    if n<=d/2:
        return n
    else:
        return n-d


def powerspectrum(array):
    """Find binned 3D power spectrum of array"""
    tarr=np.transpose(array)
    farr=np.fft.rfftn(tarr)
    powr=np.abs(farr)**2
    imax=np.size(farr[0])
    jmax=np.size(farr[1])
    kmax=np.size(farr[2])
    power=np.zeros(imax**2+jmax**2+kmax**2)
    count=np.zeros(imax**2+jmax**2+kmax**2)
    #Bin power spectrum in |k|
    for i in np.arange(0, imax):
        for j in np.arange(0, jmax):
            # The k=0 and N/2 mode need special treatment here, 
            # as they alone are not doubled. 
            #Do k=0 mode 
            kk=np.int(np.sqrt(KVAL(i,imax)**2+KVAL(j,jmax)**2))
            power[kk]+=powr[i,j,0]
            count[kk]+=1
            #Now do the k=N/2 mode 
            kk=np.int(np.sqrt(KVAL(i,imax)**2+KVAL(j,jmax)**2+KVAL(kmax/2,kmax)**2))
            power[kk]+=powr[i,j,kmax/2]
            count[kk]+=1
            #Now do the rest. Because of the symmetry, each mode counts twice
            for k in np.arange(1, kmax/2):
                kk=np.sqrt(i**2+j**2+k**2)
                power[kk]+=2*powr[i,j,k]
                count[kk]+=2
    power/=count
    return power
