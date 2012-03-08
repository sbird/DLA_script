"""A small module for computing the smoothing length of a particle simulation.
(Non-trivial in Arepo)"""

import math
import numpy as np

def get_smooth_length(bar):
    """Figures out if the particles are from AREPO or GADGET
    and computes the smoothing length.
    If we are Arepo, the smoothing length is roughly 1.5 * cell radius, where
    cell volume = 4/3 \pi (cell radius) **3 and cell volume = mass / density
    Arguments:
        Baryon particles from a simulation
    Returns:
        Array of smoothing lengths in code units.
    """
    #Are we arepo? If we are we should have this array.
    if np.any(np.array(bar.keys()) == 'Number of faces of cell'):
        rho=np.array(bar["Density"],dtype=np.float64)
        mass=np.array(bar["Masses"],dtype=np.float64)
        volume = mass/rho
        radius = (3*volume/4/math.pi)**(0.33333333)
        #Note that 4 pi/3**1/3 ~ 1.4, so the
        #geometric factors nearly cancel and the cell is almost a cube.
        hsml=1.5*radius
    else:
        #If we are gadget, the SmoothingLength array is actually the smoothing length.
        hsml=np.array(bar["SmoothingLength"],dtype=np.float64)
    return hsml
