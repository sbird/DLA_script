# -*- coding: utf-8 -*-
"""Module to compute the 2d power spectrum of a 2D field.

autofrompower_3d computes it from a 3d dimensionful power spectrum, such as you might get from CAMB

"""

import numpy as np
import math
from scipy.special import j0

def autofrompower_3d(k, pk,rr):
    """Cmpute the autocorrelation function a 3D dimensionful power spectrum, such as you might get from CAMB.
    From Challinor's structure notes:
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

