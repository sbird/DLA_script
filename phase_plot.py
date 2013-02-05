"""Module for making phase space plots of temperature vs density"""

import hdfsim
import math
import numpy as np

class MassMap:
    """Grid of mass at given density and temperature."""
    def __init__(self):
        self.minT=-8
        self.maxT=3
        self.minR=-2
        self.maxR=8
        self.map = np.zeros([300,300])
        self.r_size=np.shape(self.map)[0]
        self.t_size=np.shape(self.map)[1]

    def rho_ind(self,rho):
        """Get rho in grid units"""
        x=(rho-self.minR)/(self.maxR-self.minR)*(self.r_size-1)
        return np.array(np.around(x),dtype=int)

    def temp_ind(self,temp):
        """Get T in grid units"""
        x=(temp-self.minT)/(self.maxT-self.minT)*(self.t_size-1)
        return np.array(np.around(x),dtype=int)

    def add_to_map(self,rho, temp,mass):
        """Add data to the map"""
        self.map[self.rho_ind(rho),self.temp_ind(temp)]+=mass

    def get_rho(self):
        """Get values of rho on the grid axes"""
        return (self.maxR-self.minR)*np.arange(0,self.r_size)/(self.r_size-1)+self.minR

    def get_temp(self):
        """Get values of temp on the grid axes"""
        return (self.maxT-self.minT)*np.arange(0,self.t_size)/(self.t_size-1)+self.minT

    def get_lims(self):
        """Get limits for imshow"""
        return(self.minT,self.maxT,self.minR,self.maxR)

def get_temp_overden_mass(num,base,snap_file=0):
    """Extract from a file the temperature, rho/rho_c and mass
    for each particle.
    Outputs:
        (templog, rho/rho_c, mass)
    """
    f=hdfsim.get_file(num,base,snap_file)

    head=f["Header"].attrs
    npart=head["NumPart_ThisFile"]
    redshift=head["Redshift"]
    atime=head["Time"]
    h100=head["HubbleParam"]

    if npart[0] == 0 :
        print "No gas particles!\n"
        return

    # Baryon density parameter
    omegab0 = 0.0449
    # Scaling factors and constants
    Xh = 0.76               # Hydrogen fraction
    G = 6.672e-11           # N m^2 kg^-2
    kB = 1.3806e-23         # J K^-1
    Mpc = 3.0856e22         # m
    kpc = 3.0856e19         # m
    Msun = 1.989e30         # kg
    mH = 1.672e-27          # kg
    H0 = 1.e5/Mpc           # 100 km s^-1 Mpc^-1 in SI units
    gamma = 5.0/3.0

    rscale = (kpc * atime)/h100     # convert length to m
    #vscale = atime**0.5          # convert velocity to km s^-1
    mscale = (1e10 * Msun)/h100     # convert mass to kg
    dscale = mscale / (rscale**3.0)  # convert density to kg m^-3
    escale = 1e6            # convert energy/unit mass to J kg^-1

    bar = f["PartType0"]
    #u=escale*np.array(bar['InternalEnergy'],dtype=np.float64) # J kg^-1
    met = np.array(bar['GFM_Metallicity'], dtype=np.float64)/0.02 #solar
    ind2 = np.where(met < 1e-12)
    met[ind2] = 1e-12
    rho=dscale*np.array(bar['Density'],dtype=np.float64) # kg m^-3, ,physical
    mass=np.array(bar['Masses'],dtype=np.float64)  #1e10 Msun
    #nelec=np.array(bar['ElectronAbundance'],dtype=np.float64)
    #nH0=np.array(bar['NeutralHydrogenAbundance'],dtype=np.float64)
    f.close()
    # Convert to physical SI units. Only energy and density considered here.
    ## Mean molecular weight
    #mu = 1.0 / ((Xh * (0.75 + nelec)) + 0.25)
    #templog=np.log10(mu/kB * (gamma-1) * u * mH)
    ##### Critical matter/energy density at z=0.0
    rhoc = 3 * (H0*h100)**2 / (8. * math.pi * G) # kg m^-3
    ##### Mean hydrogen density of the Universe
    nHc = rhoc  /mH * omegab0 *Xh * (1.+redshift)**3.0
    ### Hydrogen density as a fraction of the mean hydrogen density
    overden = np.log10(rho*Xh/mH  / nHc)

    return (np.log10(met),overden,mass)

def get_temp_overden_volume(num,base,snap_file=0):
    """Extract from a file the cell radius, rho/rho_c and mass
    for each particle.
    Outputs:
        (radius, rho/rho_c, mass)
    """
    f=hdfsim.get_file(num,base,snap_file)

    head=f["Header"].attrs
    npart=head["NumPart_ThisFile"]
    redshift=head["Redshift"]
    atime=head["Time"]
    h100=head["HubbleParam"]

    if npart[0] == 0 :
        print "No gas particles!\n"
        return

    # Baryon density parameter
    omegab0 = 0.0449
    # Scaling factors and constants
    Xh = 0.76               # Hydrogen fraction
    G = 6.672e-11           # N m^2 kg^-2
    Mpc = 3.0856e22         # m
    kpc = 3.0856e19         # m
    Msun = 1.989e30         # kg
    mH = 1.672e-27          # kg
    H0 = 1.e5/Mpc           # 100 km s^-1 Mpc^-1 in SI units

    rscale = (kpc * atime)/h100     # convert length to m
    mscale = (1e10 * Msun)/h100     # convert mass to kg
    dscale = mscale / (rscale**3.0)  # convert density to kg m^-3

    bar = f["PartType0"]
    rho=dscale*np.array(bar['Density'],dtype=np.float64) # kg m^-3, ,physical
    mass=np.array(bar['Masses'],dtype=np.float64)
    #nH0=np.array(bar['NeutralHydrogenAbundance'],dtype=np.float64)

    f.close()
    # Convert to physical SI units. Only energy and density considered here.
    ##### Critical matter/energy density at z=0.0
    rhoc = 3 * (H0*h100)**2 / (8. * math.pi * G) # kg m^-3
    ##### Mean hydrogen density of the Universe
    nHc = rhoc  /mH * omegab0 *Xh * (1.+redshift)**3.0
    ### Hydrogen density as a fraction of the mean hydrogen density
    overden = np.log10(rho*Xh/mH  / nHc)
    volume=dscale*mass/rho
    rad=(3*volume/(4*math.pi))**(1/3.)
    return (rad,overden,mass)


def get_mass_map(num,base):
    """Get a map of masses gridded by temperature and density for a snapshot"""
    masses=MassMap()
    for i in np.arange(0,500) :
        try:
            (templog,overden,mass) = get_temp_overden_mass(num,base,i)
        except IOError:
            break
        ind2 = np.where((overden > masses.minR) * (overden <  masses.maxR) * (templog < masses.maxT) * (templog > masses.minT))
        masses.add_to_map(overden[ind2],templog[ind2],mass[ind2]*1e4)
    return masses

def get_volume_forest(num,base):
    """Get a map of masses gridded by cell volume and density for a snapshot"""
    masses=MassMap()
    masses.minT=10
    masses.maxT=150
    for i in np.arange(0,500) :
        try:
            (rad,overden,mass) = get_temp_overden_volume(num,base,i)
        except IOError:
            break
        ind2 = np.where((overden > -1.5) * (overden <  0.0))
        masses.add_to_map(overden[ind2],rad[ind2],mass[ind2])
        return masses

def find_particle(part_id,num,base):
    """Gets the file and position of a particle with id part_id"""
    for i in range(0,500) :
        ids=hdfsim.get_baryon_array("ParticleIDs",num,base,i,dtype=np.int64)
        ind=np.where(ids==part_id)
        if np.size(ind) > 0:
            return (ind[0][0], i)

def find_id(pos,num,base,snap_file=0):
    """Gets the id of a particle at some position"""
    ids=hdfsim.get_baryon_array("ParticleIDs",num,base,snap_file,dtype=np.int64)
    return ids[pos]

