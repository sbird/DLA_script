"""Module for finding an effective equation of state for in the Lyman-alpha forest
from a snapshot. Ported to python from Matteo Viel's IDL script."""

import h5py
import math
import numpy as np

def read_gamma(num,base):
    """Reads in an HDF5 snapshot from the NE gadget version, fits a power law to the
    equation of state for low density, low temperature gas.
    Inputs:
        num - snapshot number
        base - Snapshot directory
    Outputs:
        (T_0, \gamma) - Effective equation of state parameters
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

    N = 0

    sx = 0
    sy = 0
    sxx = 0
    sxy = 0
    met = 0
    carb = 0
    oxy = 0
    totmass=0
    totigmmass=0
    totmet = 0
    sxxm = 0
    sxym = 0
    sxm = 0
    sym = 0

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
        u=np.array(bar['InternalEnergy'],dtype=np.float64)
        rho=np.array(bar['Density'],dtype=np.float64)
        nelec=np.array(bar['ElectronAbundance'],dtype=np.float64)
        metalic = np.array(bar['GFM_Metallicity'],dtype=np.float64)
        metals = np.array(bar['GFM_Metals'],dtype=np.float64)
        mass = np.array(bar['Masses'], dtype=np.float64)
        #nH0=np.array(bar['NeutralHydrogenAbundance'])
        f.close()
        # Convert to physical SI units. Only energy and density considered here.
        rho *= dscale          # kg m^-3, ,physical
        u  *= escale         # J kg^-1

        ## Mean molecular weight
        mu = 1.0 / ((Xh * (0.75 + nelec)) + 0.25)

        #temp = mu/kB * (gamma-1) * u * mH
        #templog = alog10(temp)
        templog=np.log10(mu/kB * (gamma-1) * u * mH)

        ##### Critical matter/energy density at z=0.0
        rhoc = 3 * (H0*h100)**2 / (8. * math.pi * G) # kg m^-3

        ##### Mean hydrogen density of the Universe
        nHc = rhoc  /mH * omegab0 *Xh * (1.+redshift)**3.0

        #####  Physical hydrogen number density
        #nH = rho * Xh  / mH

        ### Hydrogen density as a fraction of the mean hydrogen density
        overden = np.log10(rho*Xh/mH  / nHc)

        ### Calculates average/median temperature in a given overdensity range#

        #overden = rho/(rhoc *omegab)

        #ind = where(overden ge -0.01 and overden le 0.01)
        #avgT0 = mean(temp(ind))
        #medT0 = median(temp(ind))
        #loT0 = min(temp(ind))
        #hiT0 = max(temp(ind))
        #
        #avgnH1 = mean(nH0(ind))
        #mednH1 = median(nH0(ind))
        #lonH1 = min(nH0(ind))
        #hinH1 = max(nH0(ind))
        #
        #print,''
        #print,'Temperature (K) at mean cosmic density'
        #print,'Average temperature [K,log]:',avgT0,alog10(avgT0)
        #print,'Median temperature [K,log]:',medT0,alog10(medT0)
        #print,'Maximum temperature [K,log]:',hiT0,alog10(hiT0)
        #print,'Minimum temperature [K,log]:',loT0,alog10(loT0)
        #
        #print
        #print,'nH1/nH at mean cosmic density'
        #print,'Mean log H1 abundance [nH1/nH,log]:',avgnH1,alog10(avgnH1)
        #print,'Median log H1 abundance [nH1/nH,log]:',mednH1,alog10(mednH1)
        #print,'Maximum log H1 abundance [nH1/nH,log]:',hinH1,alog10(hinH1)
        #print,'Minimum log H1 abundance [nH1/nH,log]:',lonH1,alog10(lonH1)
        #print
        #

        ind2 = np.where((overden > 0) * (overden <  1.5) )
        tempfit = templog[ind2]
        overdenfit = overden[ind2]

        N += np.size(ind2)
        #print, "Number of fitting points for equation of state", N

        indm = np.where(metals < 1e-10)
        metals[indm] = 1e-10
        sx += np.sum(overdenfit)
        sy += np.sum(tempfit)
        sxx += np.sum(overdenfit*overdenfit)
        sxy += np.sum(overdenfit*tempfit)
        met += np.sum(mass[ind2]*metalic[ind2])
        carb += np.sum(mass[ind2]*metals[ind2,2])
        oxy += np.sum(mass[ind2]*metals[ind2,4])
        totmet += np.sum(mass*metalic)
        totmass += np.sum(mass)
        totigmmass += np.sum(mass[ind2])
        sym += np.sum(np.log10(metals[ind2,2]))
        sxym += np.sum(overdenfit*np.log10(metals[ind2,2]))


    # log T = log(T_0) + (gamma-1) log(rho/rho_0)
    # and use least squares fit.
    delta = (N*sxx)-(sx*sx)
    a = ((sxx*sy) - (sx*sxy))/delta
    b = ((N*sxy) - (sx*sy))/delta
    amet = ((sxx*sym) - (sx*sxym))/delta
    bmet = ((N*sxym) - (sx*sym))/delta

    print num,": gamma", b+1.0,"  log(T0)", a,"  T0 (K)", (10.0)**a, "Metallicity: ", met/totigmmass,totmet/totmass, "[C/H,O/H]: ",carb/totigmmass, oxy/totigmmass,"(a_Z, b_Z): ",10**amet, bmet
    raise Exception
    return (redshift,10.0**a, b+1)

