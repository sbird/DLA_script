"""Module for finding the neutral hydrogen in a halo, applying the correction for cold clouds 
explained in Nagamine et al 2004.
    Methods:
        cold_gas_frac - Calculates the cold cloud fraction for particles from the 
                        Springel & Hernquist star formation model
        get_reproc_HI - Gets a corrected neutral hydrogen density including the above cold_gas_frac
"""

import numpy as np

def cold_gas_frac(rho, cool,rho_thresh=0.1,t_0_star=1.5,beta=0.1,T_SN=1e8,T_c = 1000):
    """Calculates the fraction of gas in cold clouds, following 
    Springel & Hernquist 2003 (astro-ph/0206393) and 
    Nagamine, Springel and Hernquist 2004 (astro-ph/0305409).

    Parameters:
        rho - Density of hot gas (g/cm^3)
        cool - cooling rate of the gas (barye/s) (1 barye = 0.1Pa, cgs pressure)
        rho_thresh - threshold density in hydrogen atoms/cm^3
               - 0.1 (Tescari & Viel)
        t_0_star - star formation timescale at threshold density
             - (MaxSfrTimescale) 1.5 in internal time units ( 1 itu ~ 0.97 Gyr)
        beta - fraction of massive stars which form supernovae (FactorSN) 0.1 in SH03.
        T_SN - Temperature of the supernova in K- 10^8 K SH03. (TempSupernova) Used to calculate u_SN
        T_c  - Temperature of the cold clouds in K- 10^3 K SH03. (TempClouds) Used to calculate u_c.
    Returns:
        The fraction of gas in cold clouds. In practice this is often 1.
    """
    #NOTE: do not modify default arguments inside the function!
    #Some constants and unit systems
    UnitLength_in_cm=3.085678e21
    UnitVelocity_in_cm_per_s=1e5
    #proton mass in g
    protonmass=1.66053886e-24
    hy_mass = 0.76 # Hydrogen massfrac
    meanweight = 4 / (1 + 3 * hy_mass)
    gamma=5./3
    boltzmann=1.38066e-16
    rho_thresh_cgs=rho_thresh*protonmass # Now in g
    t_0_star_cgs=t_0_star*(UnitLength_in_cm/UnitVelocity_in_cm_per_s) # Now in s

    #Star formation timescale
    t_star = t_0_star_cgs*np.sqrt(rho_thresh_cgs/rho)

    #SN energy: u_SN = (1-beta)/beta epsilon_SN
    u_SN =  1. / meanweight * (1.0 / (gamma -1)) * (boltzmann / protonmass) * T_SN
    #u_c - thermal energy in the cold gas.
    u_c =  1. / meanweight * (1.0 / (gamma-1)) * (boltzmann / protonmass) *T_c
    # a parameter: y = t_star \Lambda_net(\rho_h,u_h) / \rho_h (\beta u_SN - (1-\beta) u_c)
    y = t_star * cool / (rho*(beta*u_SN - (1-beta)*u_c))
    #The cold gas fraction
    f_c = 1.+ 1./(2*y) - np.sqrt(1./y+1./(4*y**2))
    return f_c

def get_reproc_rhoHI(bar,rho_thresh=0.1):
    """Get a neutral hydrogen density in cm^-2
    applying the correction in eq. 1 of Tescari & Viel
    Parameters: 
        bar = a baryon type from an HDF5 file.
        rhothresh = threshold above which to apply neutral correction"""
    #Internal gadget mass unit: 1e10 M_sun in g
    UnitMass_in_g=1.989e43
    #Internal gadget length unit: 1 kpc in cm
    UnitLength_in_cm=3.085678e21
    UnitVelocity_in_cm_per_s=1e5
    #proton mass in g
    protonmass=1.66053886e-24
    hy_mass = 0.76 # Hydrogen massfrac
    inH0=np.array(bar["NeutralHydrogenAbundance"],dtype=np.float64)
    #cgs units
    irho=np.array(bar["Density"],dtype=np.float64)*(UnitMass_in_g/UnitLength_in_cm**3)
    UnitCoolingRate_in_cgs=UnitMass_in_g*(UnitVelocity_in_cm_per_s**3/UnitLength_in_cm**4)
    icool=np.array(bar["CoolingRate"],dtype=np.float64)*UnitCoolingRate_in_cgs
    rhoH0=irho*inH0*hy_mass/protonmass
    #Default density matches Tescari & Viel and Nagamine 2004
    dens_ind=np.where(irho > protonmass*rho_thresh)
    fcold=cold_gas_frac(irho[dens_ind],icool[dens_ind],rho_thresh=rho_thresh)
    rhoH0[dens_ind]=irho[dens_ind]*fcold
    return rhoH0

