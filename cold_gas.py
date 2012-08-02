"""Module for finding the neutral hydrogen in a halo, applying the correction for cold clouds
explained in Nagamine et al 2004.
    Contains:
        StarFormation - Partially implements the star formation model of Springel & Hernquist 2003.
    Method:
        get_reproc_HI - Gets a corrected neutral hydrogen density including the fraction of gas in cold clouds
"""

import numpy as np

class StarFormation:
    """Calculates the fraction of gas in cold clouds, following
    Springel & Hernquist 2003 (astro-ph/0206393) and
    Nagamine, Springel and Hernquist 2004 (astro-ph/0305409).

    Parameters (of the star formation model):
        hubble - hubble parameter in units of 100 km/s/Mpc
        t_0_star - star formation timescale at threshold density
             - (MaxSfrTimescale) 1.5 in internal time units ( 1 itu ~ 0.97 Gyr/h)
        beta - fraction of massive stars which form supernovae (FactorSN) 0.1 in SH03.
        T_SN - Temperature of the supernova in K- 10^8 K SH03. (TempSupernova) Used to calculate u_SN
        T_c  - Temperature of the cold clouds in K- 10^3 K SH03. (TempClouds) Used to calculate u_c.
        A_0  - Supernova evaporation parameter (FactorEVP = 1000).
    """
    def __init__(self,hubble=0.7,t_0_star=1.5,beta=0.1,T_SN=1e8,T_c = 1000, A_0=1000):
        #Some constants and unit systems
        #Internal gadget mass unit: 1e10 M_sun/h in g/h
        self.UnitMass_in_g=1.989e43
        #Internal gadget length unit: 1 kpc/h in cm/h
        self.UnitLength_in_cm=3.085678e21
        #Internal velocity unit : 1 km/s in cm/s
        self.UnitVelocity_in_cm_per_s=1e5
        #proton mass in g
        self.protonmass=1.66053886e-24
        self.hy_mass = 0.76 # Hydrogen massfrac
        self.gamma=5./3
        #Boltzmann constant (cgs)
        self.boltzmann=1.38066e-16

        #Gravitational constant (cgs)
        #self.gravity = 6.672e-8

        #100 km/s/Mpc in s
        #self.h100 = 3.2407789e-18

        self.hubble=hubble

        #Supernova timescale in s
        self.t_0_star=t_0_star*(self.UnitLength_in_cm/self.UnitVelocity_in_cm_per_s)/self.hubble # Now in s

        self.beta = beta

        self.A_0 = A_0

        #u_c - thermal energy in the cold gas.
        meanweight = 4 / (1 + 3 * self.hy_mass)          #Assuming neutral gas for u_c
        self.u_c =  1. / meanweight * (1.0 / (self.gamma-1)) * (self.boltzmann / self.protonmass) *T_c

        #SN energy: u_SN = (1-beta)/beta epsilon_SN
        meanweight = 4 / (8 - 5 * (1 - self.hy_mass))    #Assuming FULL ionization for u_H
        self.u_SN =  1. / meanweight * (1.0 / (self.gamma -1)) * (self.boltzmann / self.protonmass) * T_SN


    def cold_gas_frac(self,rho, tcool,rho_thresh):
        """Calculates the fraction of gas in cold clouds, following
        Springel & Hernquist 2003 (astro-ph/0206393) and
        Nagamine, Springel and Hernquist 2004 (astro-ph/0305409).

        Parameters:
            rho - Density of hot gas (hydrogen /cm^3)
            tcool - cooling time of the gas. Zero if gas is being heated (s)
            rho_thresh - SFR threshold density in hydrogen /cm^3
        Returns:
            The fraction of gas in cold clouds. In practice this is often 1.
        """
        #Star formation timescale
        t_star = self.t_0_star*np.sqrt(rho_thresh/rho)

        #Supernova evaporation parameter
        Arho = self.A_0 * (rho_thresh/rho)**0.8

        #Internal energy in the hot phase
        u_h = self.u_SN/ (1+Arho)+self.u_c

        # a parameter: y = t_star \Lambda_net(\rho_h,u_h) / \rho_h (\beta u_SN - (1-\beta) u_c) (SH03)
        # Or in Gadget: y = t_star /t_cool * u_SN / ( \beta u_SN - (1-\beta) u_c)
        y = t_star / tcool *u_h / (self.beta*self.u_SN - (1-self.beta)*self.u_c)
        #The cold gas fraction
        f_c = 1.+ 1./(2*y) - np.sqrt(1./y+1./(4*y**2))
        return f_c

    def get_rho_thresh(self,rho_phys_thresh=0.1):
        """
        This function calculates the physical density threshold for star formation.
        It can be specified in two ways: either as a physical density threshold
        (rho_phys_thresh ) in units of hydrogen atoms per cm^3
        or as a critical density threshold (rho_crit_thresh) which is in units of rho_baryon at z=0.
        Parameters:
                rho_phys_thresh - Optional physical density threshold
                rho_crit_
        Returns:
                rho_thresh in units of g/cm^3
        """
        if rho_phys_thresh != 0:
            return rho_phys_thresh*self.protonmass #Now in g/cm^3

        u_h = self.u_SN / self.A_0

        #u_4 - thermal energy at 10^4K
        meanweight = 4 / (8 - 5 * (1 - self.hy_mass))    #Assuming FULL ionization for u_H
        u_4 =  1. / meanweight * (1.0 / (self.gamma-1)) * (self.boltzmann / self.protonmass) *1e4
        #Note: get_asymptotic_cool does not give the full answer, so do not use it.
        coolrate = self.get_asmyptotic_cool(u_h)*(self.hy_mass/self.protonmass)**2

        x = (u_h - u_4) / (u_h - self.u_c)
        return x / (1 - x)**2 * (self.beta * self.u_SN - (1 -self.beta) * self.u_c) /(self.t_0_star * coolrate)


    def get_asmyptotic_cool(self,u_h):
        """
        Get the cooling time for the asymptotically hot limit of cooling,
        where the electrons are fully ionised.
        Neglect all cooling except free-free; Gadget includes Compton from the CMB,
        but this will be negligible for high temperatures.

        Assumes no heating.
        Note: at the temperatures relevant for the threshold density, UV background excitation
        and emission is actually the dominant source of cooling.
        So this function is not useful, but I leave it here in case it is one day.
        """
        yhelium = (1 - self.hy_mass) / (4 * self.hy_mass)
        meanweight = 4 / (8 - 5 * (1 - self.hy_mass))    #Assuming FULL ionization for u_H
        temp = u_h* meanweight * (self.gamma -1) *(self.protonmass / self.boltzmann)
        print "T=",temp
        #Very hot: H and He both fully ionized
        yhelium = (1 - self.hy_mass) / (4 * self.hy_mass)
        nHp = 1.0
        nHepp = yhelium
        ne = nHp + 2.0 * nHepp

        #Free-free cooling rate
        LambdaFF = 1.42e-27 * np.sqrt(temp) * (1.1 + 0.34 * np.exp(-(5.5 - np.log(temp))**2 / 3)) * (nHp + 4 * nHepp) * ne

	    # Inverse Compton cooling off the microwave background
	    #LambdaCmptn = 5.65e-36 * ne * (temp - 2.73 * (1. + self.redshift)) * pow(1. + redshift, 4.) / nH

        return LambdaFF


    def get_reproc_rhoHI(self,bar,rho_phys_thresh=0.1):
        """Get a neutral hydrogen density in cm^-2
        applying the correction in eq. 1 of Tescari & Viel
        Parameters:
            bar = a baryon type from an HDF5 file.
            rho_phys_thresh - physical SFR threshold density in hydrogen atoms/cm^3
                            - 0.1 (Tornatore & Borgani 2007)
                            - 0.1289 (derived from the SH star formation model)
        Returns:
            nH0 - the density of neutral hydrogen in these particles in atoms/cm^3
        """
        inH0=np.array(bar["NeutralHydrogenAbundance"],dtype=np.float64)
        #Convert density to hydrogen atoms /cm^3: internal gadget density unit is h^2 (1e10 M_sun) / kpc^3
        irho=np.array(bar["Density"],dtype=np.float64)*(self.UnitMass_in_g/self.UnitLength_in_cm**3)*self.hubble**2/(self.protonmass/self.hy_mass)
        #Default density matches Tescari & Viel and Nagamine 2004
        dens_ind=np.where(irho > rho_phys_thresh)
        #UnitCoolingRate_in_cgs=UnitMass_in_g*(UnitVelocity_in_cm_per_s**3/UnitLength_in_cm**4)
        #Note: CoolingRate is really internal energy / cooling time = u / t_cool
        # HOWEVER, a CoolingRate of zero is really t_cool = 0, which is Lambda < 0, ie, heating.
        #For the star formation we are interested in y ~ t_star/t_cool,
        #So we want t_cool = InternalEnergy/CoolingRate,
        #except when CoolingRate==0, when we want t_cool = 0
        icool=np.array(bar["CoolingRate"],dtype=np.float64)
        ienergy=np.array(bar["InternalEnergy"],dtype=np.float64)
        cool=icool[dens_ind]
        ind=np.where(cool == 0)
        #Set cool to a very large number to avoid divide by zero
        cool[ind]=1e99
        tcool = ienergy[dens_ind]/cool
        #Convert from internal time units, normally 9.8x10^8 yr/h to s.
        tcool *= (self.UnitLength_in_cm/self.UnitVelocity_in_cm_per_s)/self.hubble # Now in s
        fcold=self.cold_gas_frac(irho[dens_ind],tcool,rho_phys_thresh)
        #Adjust the neutral hydrogen fraction
        inH0[dens_ind]=fcold

        #Calculate rho_HI
        nH0=irho*inH0
        #Now in atoms /cm^3
        return nH0

    def get_reproc_rhoHI(self,bar,rho_phys_thresh=6.3e-3):
        """Get a neutral hydrogen density with a self-shielding correction as suggested by Yajima Nagamine 2012 (1112.5691)
        This is just neutral over a certain density."""
        inH0=np.array(bar["NeutralHydrogenAbundance"],dtype=np.float64)
        #Convert density to hydrogen atoms /cm^3: internal gadget density unit is h^2 (1e10 M_sun) / kpc^3
        irho=np.array(bar["Density"],dtype=np.float64)*(self.UnitMass_in_g/self.UnitLength_in_cm**3)*self.hubble**2/(self.protonmass/self.hy_mass)
        #Slightly less sharp cutoff power law fit to data
        r2 = 10**-2.3437
        r1 = 10**-1.81844
        dens_ind=np.where(irho > r1)
        inH0[dens_ind]=1.
        ind2 = np.where((irho < r1)*(irho > r2))
        #Interpolate between r1 and r2
        n=2.6851
        inH0[ind2] = (inH0[ind2]*(r1-irho[ind2])**n+(irho[ind2]-r2)**n)/(r1-r2)**n
        #Calculate rho_HI
        nH0=irho*inH0
        #Now in atoms /cm^3
        return nH0
