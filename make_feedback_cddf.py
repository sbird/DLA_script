#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make some plots of the CDDF from the cosmo runs"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

import dla_plots as dp
import dla_data
import os.path as path
import myname
import numpy as np
from save_figure import save_figure

outdir = myname.base + "plots/grid"

#Colors and linestyles for the simulations
colors = {0:"red", 1:"purple", 2:"blue", 3:"green", 4:"orange", 5:"cyan"}
lss = {0:"--",1:":", 2:"-",3:"-.", 4:"-", 5:"--"}

def plot_cddf_a_halo(sim, snap, ff=True, moment=False):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10)
    ahalo.plot_column_density(color=colors[sim], ls=lss[sim], moment=moment)
    del ahalo

def plot_H2_effect(sim, snap):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, True)
    savefile = path.join(halo,"snapdir_"+str(snap).rjust(3,'0'),"boxhi_grid.hdf5")
    ahalo = dp.PrettyBox(halo, snap, nslice=10, savefile=savefile)
    ahalo.plot_column_density(color="blue", ls="--")
    del ahalo
    savefile = path.join(halo,"snapdir_"+str(snap).rjust(3,'0'),"boxhi_grid_H2.hdf5")
    ahalo = dp.PrettyBox(halo, snap, nslice=10, savefile=savefile)
    ahalo.plot_column_density(color="red")
    dla_data.column_density_data()
    del ahalo
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_H2_"+str(snap)))
    plt.clf()

def plot_UVB_effect():
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(5, True)
    savefile = path.join(halo,"snapdir_003/boxhi_grid_rahHI.hdf5")
    ahalo = dp.PrettyBox(halo, 3, nslice=10, savefile=savefile)
    ahalo.plot_column_density(color="blue", ls="--", moment=True)
    savefile = path.join(halo,"snapdir_003/boxhi_grid_rahHI_doub_UVB.hdf5")
    ahalo = dp.PrettyBox(halo, 3, nslice=10, savefile=savefile)
    ahalo.plot_column_density(color="red",moment=True)
    dla_data.column_density_data(moment=True)
    save_figure(path.join(outdir, "cosmo5_UVB_3"))
    plt.clf()

def plot_grid_res():
    """The effect of a finer grid"""
    halo = myname.get_name(5, True)
    savefile = path.join(halo,"snapdir_003/boxhi_grid_10240.hdf5")
    ahalo = dp.PrettyBox(halo, 3, nslice=10, savefile=savefile)
    ahalo.plot_column_density(color="blue", ls="--", moment=True)
#     savefile = path.join(halo,"snapdir_003/boxhi_grid_16384.hdf5")
    ahalo2 = dp.PrettyBox(halo, 3, nslice=10)

    ahalo2.plot_column_density(color="red",moment=True, ls="-.")
    dla_data.column_density_data(moment=True)
    save_figure(path.join(outdir, "cosmo5_grid_3"))
    plt.clf()

def plot_covering_frac(sim, snap, ff=True):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10)
    ahalo.plot_sigma_DLA()
    del ahalo
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_covering_z"+str(snap)))
    plt.clf()

def plot_halohist(snap):
    """Plot a histogram of nearby halos"""
    for sim in xrange(5):
        halo = myname.get_name(sim, True)
        ahalo = dp.PrettyBox(halo, snap, nslice=10)
        ahalo.plot_halo_hist(color=colors[sim])
        del ahalo
    save_figure(path.join(outdir, "cosmo_halohist_z"+str(snap)))
    plt.clf()

def get_rhohi_dndx(sim, ff=True):
    """Plot rho_HI and dndx across redshift"""
    halo = myname.get_name(sim, ff)
    ss = {4:1, 3.5:2, 3:3, 2.5:4, 2:5}
    dndx=[]
    rhohi=[]
    zzz = []
    for zz in (4, 3, 2):
        try:
            ahalo = dp.PrettyBox(halo, ss[zz], nslice=10)
            dndx.append(ahalo.line_density())
            rhohi.append(ahalo.omega_DLA())
            zzz.append(zz)
            del ahalo
        except IOError:
            continue
    return (zzz, dndx, rhohi)


def plot_rel_cddf(snap):
    """Load and make a plot of the difference between two simulations"""
    basen = myname.get_name(1)
    base = dp.PrettyBox(basen, snap, nslice=10)
    cddf_base = base.column_density_function()
    for xx in (0,2,3,4,5):
        halo2 = myname.get_name(xx)
        ahalo2 = dp.PrettyBox(halo2, snap, nslice=10)
        cddf = ahalo2.column_density_function()
        plt.semilogx(cddf_base[0], np.log10(cddf[1]/cddf_base[1]), color=colors[xx], ls=lss[xx])
    plt.ylim(-0.25,0.5)
    save_figure(path.join(outdir,"cosmo_rel_cddf_z3"))
    plt.clf()

def plot_all_rho():
    """Make the rho_HI plot with labels etc"""
    #Cosmo0
    for i in xrange(6):
        (zzz,dndx,omegadla) = get_rhohi_dndx(i)
        plt.figure(1)
        plt.plot(zzz,dndx, color=colors[i], ls=lss[i])
        plt.figure(2)
        plt.plot(zzz,omegadla, color=colors[i], ls=lss[i])
    plt.figure(1)
    plt.xlabel("z")
    plt.ylabel(r"$dN / dX$")
    dla_data.dndx_not()
    dla_data.dndx_pro()
    save_figure(path.join(outdir,"cosmo_dndx"))
    plt.clf()

    plt.figure(2)
    plt.xlabel("z")
    plt.ylabel(r"$10^3 \Omega_\mathrm{DLA}$")
    dla_data.omegahi_not()
    dla_data.omegahi_pro()
    save_figure(path.join(outdir,"cosmo_rhohi"))
    plt.clf()

def plot_breakdown():
    """Make a plot of the column density function, broken down by halo mass."""
    for sss in (0,2,3):
        halo = myname.get_name(sss, True)
        ahalo = dp.PrettyBox(halo, 3, nslice=10)
        ahalo.plot_column_density_breakdown(color=colors[sss], minN=20., maxN=22.5)
        del ahalo

    save_figure(path.join(outdir,"cosmo_cddf_break"))
    plt.clf()

if __name__ == "__main__":
#     plot_H2_effect(2,3)
    plot_rel_cddf(3)
    plot_UVB_effect()
    plot_all_rho()
    #Make a plot of the column density functions.
    for ss in xrange(6):
        plot_cddf_a_halo(ss, 3)

    dla_data.column_density_data()

    save_figure(path.join(outdir,"cosmo_cddf_z3"))
    plt.clf()

    #Plot first moment
    for ss in xrange(6):
        plot_cddf_a_halo(ss, 3, moment=True)

    dla_data.column_density_data(moment=True)

    save_figure(path.join(outdir,"cosmo_cddf_z3_moment"))
    plt.clf()

#     plot_breakdown()
    #A plot of the redshift evolution
    # zz = [1,3,5]
    # for ii in (0,1,2):
    #     plot_cddf_a_halo(2, zz[ii], color=colors[ii], ff=True)
    #
    # dla_data.column_density_data()
    # plt.title("Column density function at z=4-2")
    # save_figure(path.join(outdir,"cosmo0_cddf_zz"))
    # plt.clf()
    #
#     plot_halohist(3)

    #Make a plot of the effect of AGN on the cddf.
    for ss in (2,1):
        plot_cddf_a_halo(ss, 5, moment=True)
    dla_data.column_density_data(moment=True)

    save_figure(path.join(outdir,"cosmo_cddf_agn_z2"))
    plt.clf()

# plot_rel_cddf(3,0,3)
# plot_rel_cddf(2,0,3)
# plot_rel_cddf(2,0,68, color="grey")
# plt.xlim(1e18, 1e22)
# plt.ylim(0.8,1)
# save_figure(path.join(outdir,"cosmo_rel_cddf_z3"))
# plt.clf()
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
#plot_metal_halo(0,3,15)
#plt.xlim(-305,305)
#plt.ylim(-305, 305)
#save_figure(path.join(outdir,"cosmo0_metal_halo_z3"))
#plt.clf()
#
#plot_metal_halo(3,3,15)
#plt.xlim(-305,305)
#plt.ylim(-305, 305)
#save_figure(path.join(outdir,"cosmo3_metal_halo_z3"))
#plt.clf()
#
#plot_halo(0,3,15)
#plt.xlim(-305,305)
#plt.ylim(-305, 305)
#save_figure(path.join(outdir,"cosmo0_halo_z3"))
#plt.clf()
#
