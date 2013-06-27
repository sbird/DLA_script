# -*- coding: utf-8 -*-
"""Make a plot showing the fraction of SiII with density and also the fraction of HI with density"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

import convert_cloudy as cc
import cold_gas as cg
import os.path as path
import numpy as np
from save_figure import save_figure

base="/home/spb/scratch/Cosmo/"
outdir = base + "plots/"
print "Plots at: ",outdir

def plot_SivsHI(mets=0.05):
    """
        Plot the SiII fraction as a function of density, for some metallicity.
        Mets is an array, metallicity as a fraction of solar.
    """
    if np.size(mets) == 1:
        mets = np.array([mets,])
    tab = cc.CloudyTable(3)

    #The hydrogen density in atoms/cm^3
    dens = np.logspace(-5,2,100)

    #Roughly mean DLA metallicity

    tabHI = cg.RahmatiRT(3, 0.71)

    tempHI = 1e4*np.ones_like(dens)
    fracHI = tabHI.neutral_fraction(dens,tempHI)
    plt.semilogx(dens, fracHI, color="red",ls="--")

    ls = [":","-","-."]
    for met in mets:
        metSi = tab.get_solar("Si")*met*np.ones_like(dens)
        fracSi = tab.ion("Si",2,metSi,dens)
        plt.semilogx(dens, fracSi, color="green",ls=ls.pop())

    plt.xlabel(r"$\rho_\mathrm{H}\; (\mathrm{amu}/\mathrm{cm}^3$)")
    plt.ylabel(r"$\mathrm{m}_\mathrm{SiII} / \mathrm{m}_\mathrm{Si}$")
    plt.show()
    save_figure(path.join(outdir,"Si_fracs"))

#Values are mean fit for 2e9, 1e10 and 1e11 halos resp.
plot_SivsHI([2.5e-3,1e-2,6e-2])
