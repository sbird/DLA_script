"""Script for plotting DLA data; ie, for the column density function, etc.
Column density data courtesy of Yajima Hidenobu"""

import matplotlib.pyplot as plt
import os.path as path
import numpy as np

def column_density_data(datadir="data", moment=False):
    """Plot the data on the column density function at z=3"""
#     celine_data(datadir)
#     peroux_data(datadir)
    omeara_data(datadir, moment)
    noterdaeme_12_data(datadir, moment)
#     prochaska_data(datadir)
#     prochaska_05_data(datadir)
    prochaska_10_data(datadir, moment)

def format_error_bars(data):
    """Take a file formatted for SuperMongo and format it instead for matplotlib,
    adjusting the format of the error bars
    In-Format: log10(x,y,x-errorx,x+errorx, y-errorx, y+errory)
    Out-Format: x,y,lerrorx,uerrorx,lerrory,uerrory
    """
    data = 10**data
    #Format the error bars the way matplotlib likes them
    data[:,2]=-data[:,2]+data[:,0]
    data[:,3]=data[:,3]-data[:,0]
    data[:,4]=-data[:,4]+data[:,1]
    data[:,5]=data[:,5]-data[:,1]
    return data

def celine_data(datadir="data"):
    """Plot the Peroux 2001 data on the column density function at z=3"""
    celine=np.loadtxt(path.join(datadir,"fn_celine_z3.dat"))
    celine=format_error_bars(celine)
    plt.errorbar(celine[:,0],celine[:,1],xerr=[celine[:,2],celine[:,3]], yerr=[celine[:,4],celine[:,5]], fmt='o')
    return

def peroux_data(datadir="data"):
    """Plot the Peroux 2005 data on the column density function at z=3"""
    peroux=np.loadtxt(path.join(datadir,"peroux05_z3.dat"))
    peroux=format_error_bars(peroux)
    plt.errorbar(peroux[:,0],peroux[:,1],xerr=[peroux[:,2],peroux[:,3]], yerr=[peroux[:,4],peroux[:,5]], fmt='*')

def omeara_data(datadir="data", moment=False):
    """Plot the O'Meara 07 data on the column density function (LLS). Mean redshift is 3.1"""
    omera=np.loadtxt(path.join(datadir,"summary.dat"))
    omera=format_error_bars(omera)
    #Take the first moment
    if moment:
        for i in (1,4,5):
            omera[:,i]*=omera[:,0]
    plt.errorbar(omera[:,0],omera[:,1],xerr=[omera[:,2],omera[:,3]], yerr=[omera[:,4],omera[:,5]], fmt='s',color='black',ms=8)

def noterdaeme_data(datadir="data"):
    """Plot the Noterdaeme 09 data on the column density function at z=2-3
    Format: x, y, xerr, yerr (in logspace)"""
    data=np.loadtxt(path.join(datadir,"fhix.dat"))
    #Madness to put log errors into non-log
    uxer=10**(data[:,2]+data[:,0])-10**data[:,0]
    lxer=-10**(-data[:,2]+data[:,0])+10**data[:,0]
    uyer=10**(data[:,3]+data[:,1])-10**data[:,1]
    lyer=-10**(-data[:,3]+data[:,1])+10**data[:,1]
    NHI = 10**data[:,0]
    plt.errorbar(NHI,10**data[:,1]*NHI,xerr=[lxer,uxer],yerr=[lyer*NHI,uyer*NHI], fmt='^',color='green',ms=10)

def noterdaeme_12_data(datadir="data", moment=False):
    """Plot the Noterdaeme 12 data (1210.1213) on the column density function at z=2-3.5
    Format: x, y, xerr, yerr (in logspace)"""
    data=np.loadtxt(path.join(datadir,"not_2012.dat"))
    #Madness to put log errors into non-log
    uxer=10**(data[:,2]+data[:,0])-10**data[:,0]
    lxer=-10**(-data[:,2]+data[:,0])+10**data[:,0]
    uyer=10**(data[:,3]+data[:,1])-10**data[:,1]
    lyer=-10**(-data[:,3]+data[:,1])+10**data[:,1]
    NHI = 10**data[:,0]
    cddf = 10**data[:,1]
    if moment:
        lyer*=NHI
        uyer*=NHI
        cddf*=NHI
    plt.errorbar(NHI,cddf,xerr=[lxer,uxer],yerr=[lyer,uyer], fmt='^',color='green',ms=10)

def prochaska_data(datadir="data"):
    """Plot the Prochaska and Wolfe 10 data on the column density function.
    Mean redshift is 3.05.
    Format: x lowerxerr upperxerr y"""
    data=np.loadtxt(path.join(datadir,"2fn_sdss_dr5.dat"))
    data=10**data
    plt.errorbar(data[:,0],data[:,3],xerr=[data[:,0]-data[:,1],data[:,2]-data[:,0]],fmt='.')

def prochaska_05_data(datadir="data"):
    """Plot the Prochaska 05 data on the column density function at z=3"""
    omera=np.loadtxt(path.join(datadir,"prochaska_05.dat"))
    omera=format_error_bars(omera)
    plt.errorbar(omera[:,0],omera[:,1],xerr=[omera[:,2],omera[:,3]], yerr=[omera[:,4],omera[:,5]], fmt='D')

def prochaska_10_data(datadir="data", moment=False):
    """Plot the LLS only data of Prochaska 2010, given as a box rather than the more conventional error bars.
    This is at z=3.7"""
    data=np.loadtxt(path.join(datadir,"prochaska_lls.dat"))
    ax=plt.gca()
    cddf = data[:,1]
    if moment:
        cddf+=data[:,0]
    ax.fill(10.**data[:,0],10.**cddf,'grey')


def dndx_pro(datadir="data"):
    """Plot the line densities for DLAs from Prochaska & Wolfe 2009, 0811.2003"""
    data = np.loadtxt(path.join(datadir,"dndx.txt"))
    zcen = (data[1:-1,0]+data[1:-1,1])/2.
    plt.errorbar(zcen, data[1:-1,2],xerr=[zcen-data[1:-1,0], data[1:-1,1]-zcen], yerr=data[1:-1,3], fmt="o",color="orange")

def omegahi_pro(datadir="data"):
    """Plot the total rho_HI density for DLAs from Prochaska & Wolfe 2009, 0811.2003"""
    data = np.loadtxt(path.join(datadir,"dndx.txt"))
    zcen = (data[1:-1,0]+data[1:-1,1])/2.
    rhohi = data[1:-1,4]
    #This is rho_crit at z=0
    rho_crit = 9.3125685124148235e-30
    #This converts from 1e8 M_sun/Mpc^3 to g/cm^3
    conv = 6.7699111782945424e-33
    #A factor of 0.76 from HI mass to gas mass
    #Note: this factor is 0.74, so that the Noterdaeme
    #Omega_DLA is numerically similar to the rho_HI of Prochaska
    omega_DLA = rhohi*conv/rho_crit*1000
    plt.errorbar(zcen, omega_DLA,xerr=[zcen-data[1:-1,0], data[1:-1,1]-zcen], yerr=data[1:-1,5], fmt="o",color="orange")

def omegahi_not():
    """Omega_DLA from Noterdaeme 2012, 1210.1213"""
    #He divides these measurements by 0.76,
    #which he thinks gives him the neutral gas mass in DLAs, because all this hydrogen
    #is neutral. However, some of the hydrogen is molecular, so the factor is daft.
    omega_dla = np.array([0.99, 0.87, 1.04, 1.1, 1.27])*0.76
    omega_err = np.array([0.05,0.04, 0.05,0.08,0.13])
    zz = [2.15,2.45,2.75,3.05,3.35]
    plt.errorbar(zz, omega_dla,xerr=0.15, yerr=omega_err, fmt="s",color="black")

def dndx_not():
    """dNdX from Noterdaeme 2012, 1210.1213"""
    #No error on dndz...use the systematic correction.
    dndz = np.array([0.2,0.2,0.25,0.29,0.36])
    #No error bars quoted in the paper for dndz (?)
    zz = [2.15,2.45,2.75,3.05,3.35]
    dzdx = np.array([3690/11625.,4509/14841.,2867/9900.,1620/5834.,789/2883.])
    plt.errorbar(zz,dndz*dzdx,xerr=0.15,fmt="s",color="black")

# def prochaska_10_data():
#     """Plot the six-power-law model of Prochaska 2010. A little too complicated."""
#     NHI = np.logspace(14.5,23)
#     fN = np.array(NHI)
#     #Central power law parameters: break, exponent, norm
#     #Lya, pLLS, LLS, SLLS, DLA1, DLA2
#     #Note DLA is defined as k*(N/10^21.75)^b,
#     #but all the others are k * N^b...
#     breaks=np.array([14.5,17.3,19.,20.3,21.75])
#     exp = np.array([-1.5,-1.9,-0.8,-1.2,-1.8,-3])
#     norm = np.array([, ,10.**(-4.5) , 4.56e6,7e-25,7e-25])
#     #DLAs
#     ind = np.where(NHI > 10**breaks[3])
#
#

def absorption_distance():
    """Compute X(z), the absorption distance per sightline (eq. 9 of Nagamine et al 2003)
    in dimensionless units."""
    #h * 100 km/s/Mpc in h/s
    h100=3.2407789e-18
    # in cm/s
    light=2.9979e10
    #Internal gadget length unit: 1 kpc/h in cm/h
    UnitLength_in_cm=3.085678e21
    redshift = 3
    box = 25000
    #Units: h/s   s/cm                        kpc/h      cm/kpc
    return h100/light*(1+redshift)**2*box*UnitLength_in_cm

def altay_data():
    """Plot the simulation cddf from Altay 2011: 1012.4014"""
    #His bins
    edges = 10**np.arange(17,22.51,0.1)
    #His data values
    cddf = np.array([858492, 747955, 658685, 582018, 518006, 468662, 431614, 406575, 387631, 374532, 359789, 350348, 342146, 334534, 329178, 324411, 320648, 318207, 316232, 314852, 314504, 314583, 313942, 315802, 316330, 316884, 317336, 317979, 316526, 317212, 314774,  309333,  302340,  291816,  275818,  254368,  228520,  198641,  167671,  135412,  103583, 76751, 54326, 37745, 25140, 16784, 10938, 6740, 3667, 1614, 637, 206, 33, 14, 7])
    center = np.array([(edges[i]+edges[i+1])/2. for i in range(0,np.size(edges)-1)])
    width =  np.array([edges[i+1]-edges[i] for i in range(0,np.size(edges)-1)])
    #Grid size (in cm^2)
    dX=absorption_distance()
    tot_cells = 16384**2
    cddf=(cddf)/(width*dX*tot_cells)
    plt.loglog(center,cddf)


