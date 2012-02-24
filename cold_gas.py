import numpy as np

def cold_gas_frac(rho,u, cool,rho_thresh=1.66053886e-25,A_0 = 1000,t_0_star=2.1,beta=0.1,epsilon_SN=4e48):
        """Calculates the fraction of gas in cold clouds, following 
        Springel & Hernquist 2003 (astro-ph/0206393) and 
        Nagamine, Springel and Hernquist 2004 (astro-ph/0305409).

        Parameters:
                rho - Density of hot gas (g/cm^3)
                u - Thermal energy of hot gas (ergs/g = (cm/s)^2)
                cool - cooling rate of the gas (barye/s) (1 barye = 0.1Pa, cgs pressure)
                rho_thresh - threshold density in g/cm^3. Set to 0.1 in units of hydrogen atoms/cm^3 
                                following Tescari & Viel
                A_0 - supernova evaporation parameter - 1000 (SH03)
                t_0_star - star formation timescale at threshold density
                         - 2.1 Gyr in SH03
                beta - fraction of massive stars which form supernovae 0.1 in SH03.
                epsilon_SN - energy deposited from each supernova 
                           - 4x10^48 ergs/M_sun in SH03
        Returns:
                The fraction of gas in cold clouds. In practice this is often 1.
        """
        t_0_star*=31556926*1e9 # Now in s
        epsilon_SN/=1.989e33 # Now in ergs/g
        #Star formation timescale
        t_star = t_0_star*np.sqrt(rho_thresh/rho)
        #SN energy scale: u_SN = (1-beta)/beta epsilon_SN
        u_SN = (1-beta)*epsilon_SN/beta
        #supernova evaporation parameter A
        A = A_0 *(rho/rho_thresh)**(-0.8)
        #u_c - thermal energy in the cold gas. Equilibrium at:
        u_c = u - u_SN/(A+1.)
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
        inH0=np.array(bar["NeutralHydrogenAbundance"],dtype=np.float64)
        #cgs units
        irho=np.array(bar["Density"],dtype=np.float64)*(UnitMass_in_g/UnitLength_in_cm**3)
        iu=np.array(bar["InternalEnergy"],dtype=np.float64)*UnitVelocity_in_cm_per_s**2
        UnitCoolingRate_in_cgs=UnitMass_in_g*(UnitVelocity_in_cm_per_s**3/UnitLength_in_cm**4)
        icool=np.array(bar["CoolingRate"],dtype=np.float64)*UnitCoolingRate_in_cgs
        rhoH0=irho*inH0/protonmass
        #Default density matches Tescari & Viel and Nagamine 2004
        dens_ind=np.where(irho > protonmass*rho_thresh)
        fcold=cold_gas_frac(irho[dens_ind],iu[dens_ind],icool[dens_ind],rho_thresh=rho_thresh*protonmass)
        rhoH0[dens_ind]=irho[dens_ind]*fcold
        return rhoH0

