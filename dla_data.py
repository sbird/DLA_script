"""Script for plotting DLA data; ie, for the column density function, etc.
Column density data courtesy of Yajima Hidenobu"""

import matplotlib.pyplot as plt
import os.path as path
import numpy as np

def column_density_data(datadir="data"):
    """Plot the data on the column density function at z=3"""
#     celine_data(datadir)
#     peroux_data(datadir)
    omeara_data(datadir)
    noterdaeme_data(datadir)
#     prochaska_data(datadir)
#     prochaska_05_data(datadir)
    prochaska_10_data(datadir)

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

def omeara_data(datadir="data"):
    """Plot the O'Meara 07 data on the column density function (LLS). Mean redshift is 3.1"""
    omera=np.loadtxt(path.join(datadir,"summary.dat"))
    omera=format_error_bars(omera)
    plt.errorbar(omera[:,0],omera[:,1],xerr=[omera[:,2],omera[:,3]], yerr=[omera[:,4],omera[:,5]], fmt='s',color='black')

def noterdaeme_data(datadir="data"):
    """Plot the Noterdaeme 09 data on the column density function at z=2-3
    Format: x, y, xerr, yerr (in logspace)"""
    data=np.loadtxt(path.join(datadir,"fhix.dat"))
    #Madness to put log errors into non-log
    uxer=10**(data[:,2]+data[:,0])-10**data[:,0]
    lxer=-10**(-data[:,2]+data[:,0])+10**data[:,0]
    uyer=10**(data[:,3]+data[:,1])-10**data[:,1]
    lyer=-10**(-data[:,3]+data[:,1])+10**data[:,1]
    plt.errorbar(10**data[:,0],10**data[:,1],xerr=[lxer,uxer],yerr=[lyer,uyer], fmt='^',color='green')

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

def prochaska_10_data(datadir="data"):
    """Plot the LLS only data of Prochaska 2010, given as a box rather than the more conventional error bars.
    This is at z=3.7"""
    data=np.loadtxt(path.join(datadir,"prochaska_lls.dat"))
    ax=plt.gca()
    ax.fill(10.**data[:,0],10.**data[:,1],'grey')


# def prochaska_10_data():
#     """Plot the six-power-law model of Prochaska 2010. A little too complicated.
#         Also I don't understand wtf he's doing. He produces the NH > 21.75 power law apparently from
#       one point in P&W 10, he never gives the parameters of his power laws, his fit makes no sense,
#       and he gives a mean redhift of 3.7, despite all his data coming from z=3.1 (??)
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
