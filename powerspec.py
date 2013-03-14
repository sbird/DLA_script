# -*- coding: utf-8 -*-
"""Module to compute the 2d power spectrum of a 2D field.

powerspectrum_2d computes the power spectrum
autofromdelta computes the autocorrelation function from the output of powerspectrum_2d.
autofrompower_3d computes it from a 3d dimensionful power spectrum, such as you might get from CAMB

"""

import numpy as np
import _power_priv
import math
from scipy.special import j0

def powerspectrum_2d(field, box):
    """Compute the powerspectrum of a 2d field using fftw.
       Fourier convention used is that of CAMB.
       Arguments: field - the field
           box - size of the field, in whatever units you want.
       Outputs:
           (keffs, power, count) in the same units as the box
           power is the dimensionless power spectrum: P(k) * k^2 /(2π^2)
           Note the zeroth mode contains the DC component, usually zero.
    """
    nrbins=int(np.floor(np.sqrt(2)*((np.shape(field)[0]+1.0)/2.0)+1))
    (keffs, power, count) = _power_priv.powerspectrum_2d(field,nrbins)
    #Convert units:
    power*=keffs**2
    keffs/=box
    #Convert Fourier convention from FFTW to CAMB: factor is (2π)^2 because we are 2d
    power/=4*math.pi/(2*math.pi)**2
    return (keffs, power, count)

def autofrompower_3d(k, pk,rr):
    """From Challinor's structure notes.
        P(k) =  < δ δ*>
        Δ^2 = P(k) k^3/(2π^2)
        ζ(r) = int dk/k Δ^2 j_0(kr)
             = int dk (k^2) P(k) j_0(kr) / (2π^2)
        Arguments:
            k - k values
            pk - power spectrum
            r - values of r = | x-y |
                at which to evaluate the autocorrelation
    """
    auto = np.array([np.sum(pk*j0(k*r)*k**2/2/math.pi**2)/np.size(k) for r in rr])
    return auto

def autofromdelta(k, delta,rr):
    """From Challinor's structure notes.
        P(k) =  < δ δ*>
        Δ^2 = P(k) k^3/(2π^2)
        ζ(r) = int dk/k Δ^2 j_0(kr)
        Arguments:
            k - k values
            delta - dimensionless power spectrum
            r - values of r = | x-y |
                at which to evaluate the autocorrelation
    """
    auto = np.array([np.sum(delta*j0(k*r)/k)/np.size(k) for r in rr])
    return auto

