"""Module for finding the mean mass-weighted metallicity at a given density."""

import h5py
import math
import numpy as np

def get_met(num,base):
    """Reads in an HDF5 snapshot and finds the mass-weighted average metallicity.
    """
    # Baryon density parameter
    omegab0 = 0.0449
    singlefile=False
    #base="/home/spb41/data2/runs/bf2/"
    snap=str(num).rjust(3,'0')
    fname=base+"/snapdir_"+snap+"/snap_"+snap
    try:
        f=h5py.File(fname+".0.hdf5",'r')
    except IOError:
        fname=base+"/snap_"+snap
        f=h5py.File(fname+".hdf5",'r')
        singlefile=True

    print 'Reading file from:',fname

    head=f["Header"].attrs
    npart=head["NumPart_ThisFile"]
    redshift=head["Redshift"]
    print "z=",redshift
    atime=head["Time"]
    h100=head["HubbleParam"]

    if npart[0] == 0 :
        print "No gas particles!\n"
        return

    f.close()

    # Scaling factors and constants
    Xh = 0.76               # Hydrogen fraction
    G = 6.672e-11           # N m^2 kg^-2
    Mpc = 3.0856e22         # m
    kpc = 3.0856e19         # m
    Msun = 1.989e30         # kg
    mH = 1.672e-27          # kg
    H0 = 1.e5/Mpc           # 100 km s^-1 Mpc^-1 in SI units

    rscale = (kpc * atime)/h100     # convert length to m
    #vscale = atime**0.5          # convert velocity to km s^-1
    mscale = (1e10 * Msun)/h100     # convert mass to kg
    dscale = mscale / (rscale**3.0)  # convert density to kg m^-3

    dedges = np.logspace(-2, 5,51)
    denbins = np.array([(dedges[i]+dedges[i+1])/2. for i in range(0,np.size(dedges)-1)])
    nbins = np.size(denbins)
    totmass = np.zeros(nbins)
    totmet = np.zeros(nbins)

    for i in np.arange(0,500) :
        ffname=fname+"."+str(i)+".hdf5"
        if singlefile:
            ffname=fname+".hdf5"
            if i > 0:
                break
        #print 'Reading file ',ffname
        try:
            f=h5py.File(ffname,'r')
        except IOError:
            break
        head=f["Header"].attrs
        npart=head["NumPart_ThisFile"]
        if npart[0] == 0 :
            print "No gas particles in file ",i,"!\n"
            break
        bar = f["PartType0"]
        rho=np.array(bar['Density'],dtype=np.float64)
        metalic = np.array(bar['GFM_Metallicity'],dtype=np.float64)
        mass = np.array(bar['Masses'], dtype=np.float64)
        #metals = np.array(bar['GFM_Metals'],dtype=np.float64)
        f.close()
        # Convert to physical SI units. Only energy and density considered here.
        rho *= dscale          # kg m^-3, ,physical

        ##### Critical matter/energy density at z=0.0
        rhoc = 3 * (H0*h100)**2 / (8. * math.pi * G) # kg m^-3

        ##### Mean hydrogen density of the Universe
        nHc = rhoc  /mH * omegab0 *Xh * (1.+redshift)**3.0

        #####  Physical hydrogen number density
        #nH = rho * Xh  / mH

        ### Hydrogen density as a fraction of the mean hydrogen density
        overden = rho*Xh/mH  / nHc
        print np.max(overden)

        for j in np.arange(0,np.size(denbins)):
            ind = np.where(np.logical_and(overden > dedges[j], overden < dedges[j+1]))
            totmass[j] += np.sum(mass[ind])
            totmet[j] += np.sum(mass[ind]*metalic[ind])

    ind = np.where(totmass > 0)
    totmet[ind]/=totmass[ind]

    return (denbins, totmet/0.0188)
