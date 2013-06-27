#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make some plots of the CDDF from the cosmo runs"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

import dla_plots as dp
import dla_data
import os.path as path
from save_figure import save_figure

base="/home/spb/scratch/Cosmo/"
outdir = base + "plots/grid"

#Colors and linestyles for the simulations
colors={ 0:"red", 1:"orange",2:"blue", 3:"green"}
lss= {0:"--",1:":",2:"-",3:"-."}

def plot_cddf_a_halo(sim, snap, ff=True):
    """Load a simulation and plot its cddf"""
    halo = "Cosmo"+str(sim)+"_V6"
    savefile = None
    if ff:
        halo+="_512"
        savefile = path.join(base+halo,"snapdir_"+str(snap).rjust(3,'0'),"boxhi_grid_H2.hdf5")
    ahalo = dp.PrettyBox(base+halo, snap, nslice=10, savefile=savefile)
    ahalo.plot_column_density(color=colors[sim], ls=lss[sim])
    del ahalo

def plot_H2_effect(sim, snap):
    """Load a simulation and plot its cddf"""
    halo = "Cosmo"+str(sim)+"_V6_512"
    savefile = path.join(base+halo,"snapdir_"+str(snap).rjust(3,'0'),"boxhi_grid.hdf5")
    ahalo = dp.PrettyBox(base+halo, snap, nslice=10, savefile=savefile)
    ahalo.plot_column_density(color="blue", ls="--")
    del ahalo
    savefile = path.join(base+halo,"snapdir_"+str(snap).rjust(3,'0'),"boxhi_grid_H2.hdf5")
    ahalo = dp.PrettyBox(base+halo, snap, nslice=10, savefile=savefile)
    ahalo.plot_column_density(color="red")
    dla_data.column_density_data()
    del ahalo
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_H2_"+str(snap)))

def plot_covering_frac(sim, snap, ff=True):
    """Load a simulation and plot its cddf"""
    halo = "Cosmo"+str(sim)+"_V6"
    if ff:
        halo+="_512"
    ahalo = dp.PrettyBox(base+halo, snap, nslice=10)
    ahalo.plot_sigma_DLA()
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_covering_z"+str(snap)))
    plt.clf()
    ahalo.plot_halo_hist()
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_halohist_z"+str(snap)))
    del ahalo
    plt.clf()

def plot_rhohi_dndx(sim, line, ff=True):
    """Plot rho_HI and dndx across redshift"""
    halo = "Cosmo"+str(sim)+"_V6"
    if ff:
        halo+="_512"
    ss = {4:54, 3.5:57, 3:60, 2.5:64, 2:68}
    dndx=[]
    rhohi=[]
    zzz = []
    for zz in (4, 3.5, 3, 2.5,2):
        try:
            ahalo = dp.PrettyBox(base+halo, ss[zz], nslice=10)
            dndx.append(ahalo.line_density())
            rhohi.append(ahalo.rho_DLA())
            zzz.append(zz)
            del ahalo
        except IOError:
            pass
    plt.figure(1)
    plt.plot(zzz,dndx, color=colors[sim], ls=lss[sim])
    plt.figure(2)
    plt.plot(zzz,rhohi, color=colors[sim], ls=lss[sim])

def plot_rel_cddf(sim1, sim2, snap, color="black"):
    """Load and make a plot of the difference between two simulations"""
    halo1 = "Cosmo"+str(sim1)+"_V6_512"
    halo2 = "Cosmo"+str(sim2)+"_V6_512"
    ahalo1 = dp.PrettyBox(base+halo1, snap, nslice=10)
    cddf1 = ahalo1.column_density_function(0.2,16,24)
    del ahalo1
    ahalo2 = dp.PrettyBox(base+halo2, snap, nslice=10)
    cddf2 = ahalo2.column_density_function(0.2,16,24)
    plt.semilogx(cddf1[0], cddf2[1]/cddf1[1], color=color)

def plot_all_rho():
    """Make the rho_HI plot with labels etc"""
    for i in (0,2,3):
        plot_rhohi_dndx(i)
    plt.figure(1)
    plt.xlabel("z")
    plt.ylabel(r"$\frac{dN}{dX}$")
    dla_data.dndx()
    save_figure(path.join(outdir,"cosmo_dndx"))
    plt.clf()

    plt.figure(2)
    plt.xlabel("z")
    plt.ylabel(r"$\rho_\mathrm{HI}$ ($10^8 M_\odot / \mathrm{Mpc}^3$)")
    dla_data.rhohi()
    save_figure(path.join(outdir,"cosmo_rhohi"))
    plt.clf()

if __name__ == "__main__":
    # plot_H2_effect(2,60)
    plot_all_rho()
    #Make a plot of the column density functions.
    for ss in (3,2,0):
        plot_cddf_a_halo(ss, 60)

    dla_data.column_density_data()

    save_figure(path.join(outdir,"cosmo_cddf_z3"))
    plt.clf()

    #Make a plot of the column density function, broken down by halo mass.
    ahalo = dp.PrettyBox(base+"Cosmo2_V6_512", 3, nslice=10)
    ahalo.plot_column_density_breakdown()
    del ahalo
    save_figure(path.join(outdir,"cosmo2_cddf_break"))
    plt.clf()

    #A plot of the redshift evolution
    # zz = [54,60,68]
    # for ii in (0,1,2):
    #     plot_cddf_a_halo(2, zz[ii], color=colors[ii], ff=True)
    #
    # dla_data.column_density_data()
    # plt.title("Column density function at z=4-2")
    # save_figure(path.join(outdir,"cosmo0_cddf_zz"))
    # plt.clf()
    #
    # for i in (0,2,3):
    #     plot_covering_frac(i,60, True)

    #Make a plot of the effect of AGN on the cddf.
    for ss in (2,0):
        plot_cddf_a_halo(ss, 60, colors[ss], False)
    dla_data.column_density_data()

    save_figure(path.join(outdir,"cosmo_cddf_agn_z3"))
    plt.clf()

# plot_rel_cddf(3,0,60)
# plot_rel_cddf(2,0,60)
# plot_rel_cddf(2,0,68, color="grey")
# plt.xlim(1e18, 1e22)
# plt.ylim(0.8,1)
# save_figure(path.join(outdir,"cosmo_rel_cddf_z3"))
# plt.clf()
#
#def plot_cddf_breakdown(sim, snap, color="red"):
#    """Load a simulation and plot its cddf"""
#    halo = "Cosmo"+str(sim)+"_V6"
#    ahalo = dp.PrettyHalo(base+halo, snap)
#    ahalo.plot_column_density_breakdown(color=color)
#    del ahalo
#
#plot_cddf_breakdown(0, 60)
#
#plot_cddf_breakdown(3, 60, color="blue")
#plt.xlim(5e20,1e23)
#plt.ylim(5e-2,2)
#save_figure(path.join(outdir,"cosmo0_cddf_break_z3"))
#plt.clf()
#
#def plot_halo(sim, snap, num):
#    """Load a simulation and plot its cddf"""
#    halo = "Cosmo"+str(sim)+"_V6"
#    ahalo = dp.PrettyHalo(base+halo, snap)
#    ahalo.plot_pretty_halo(num)
#
#def plot_metal_halo(sim, snap, num):
#    """Load a simulation and plot its cddf"""
#    halo = "Cosmo"+str(sim)+"_V6"
#    ahalo = dp.PrettyMetal(base+halo, snap, "Si", 2)
#    ahalo.plot_pretty_halo(num)
#
#plot_metal_halo(0,60,15)
#plt.xlim(-305,305)
#plt.ylim(-305, 305)
#save_figure(path.join(outdir,"cosmo0_metal_halo_z3"))
#plt.clf()
#
#plot_metal_halo(3,60,15)
#plt.xlim(-305,305)
#plt.ylim(-305, 305)
#save_figure(path.join(outdir,"cosmo3_metal_halo_z3"))
#plt.clf()
#
#plot_halo(0,60,15)
#plt.xlim(-305,305)
#plt.ylim(-305, 305)
#save_figure(path.join(outdir,"cosmo0_halo_z3"))
#plt.clf()
#
