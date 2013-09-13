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
import vel_data
import numpy as np
from save_figure import save_figure

outdir = myname.base + "plots/grid"

#Colors and linestyles for the simulations
colors = {0:"red", 1:"purple", 2:"blue", 3:"green", 4:"orange", 5:"cyan"}
lss = {0:"--",1:":", 2:":",3:"-.", 4:"--", 5:"-"}

def plot_cddf_a_halo(sim, snap, ff=True, moment=False):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10)
    ahalo.plot_column_density(color=colors[sim], ls=lss[sim], moment=moment)
    del ahalo

def plot_metal_halo(sim, snap, ff=True, lls=False):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10)
    if lls:
        ahalo.plot_lls_metallicity(color=colors[sim], ls=lss[sim])
    else:
        ahalo.plot_dla_metallicity(color=colors[sim], ls=lss[sim])
    del ahalo

def plot_H2_effect(sim, snap):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, True)
    savefile = path.join(halo,"snapdir_"+str(snap).rjust(3,'0'),"boxhi_grid_noH2.hdf5")
    ahalo = dp.PrettyBox(halo, snap, nslice=10, savefile=savefile)
    ahalo.plot_column_density(color="blue", ls="--")
    savefile = path.join(halo,"snapdir_"+str(snap).rjust(3,'0'),"boxhi_grid_H2.hdf5")
    ahalo2 = dp.PrettyBox(halo, snap, nslice=10, savefile=savefile)
    ahalo2.plot_column_density(color="red")
    dla_data.column_density_data()
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_H2_"+str(snap)))
    plt.clf()
    cddf_base = ahalo.column_density_function()
    cddf = ahalo2.column_density_function()
    plt.semilogx(cddf_base[0], np.log10(cddf[1]/cddf_base[1]), color=colors[sim], ls=lss[sim])
    plt.ylim(-0.5,0.5)
    save_figure(path.join(outdir, "cosmo_rel"+str(sim)+"_H2_"+str(snap)))
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

def plot_cutoff():
    """Plot effect with a cutoff self-shielding"""
    halo = myname.get_name(0, True)
    savefile = path.join(halo,"snapdir_003/boxhi_grid_cutoff_H2.hdf5")
    ahalo = dp.PrettyBox(halo, 3, nslice=10, savefile=savefile)
    cutoff0 = ahalo.column_density_function()
    ahalo = dp.PrettyBox(halo, 3, nslice=10)
    normal0 = ahalo.column_density_function()
    halo = myname.get_name(1, True)
    savefile = path.join(halo,"snapdir_003/boxhi_grid_cutoff_H2.hdf5")
    ahalo = dp.PrettyBox(halo, 3, nslice=10, savefile=savefile)
    cutoff1 = ahalo.column_density_function()
    ahalo = dp.PrettyBox(halo, 3, nslice=10)
    normal1 = ahalo.column_density_function()
    plt.semilogx(cutoff0[0], np.log10(cutoff0[1]/cutoff1[1]), color="red", ls="-")
    plt.semilogx(normal0[0], np.log10(normal0[1]/normal1[1]), color="blue", ls="--")
    save_figure(path.join(outdir, "cosmo_rel_cutoff"))
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
    for sim in xrange(6):
        halo = myname.get_name(sim, True)
        ahalo = dp.PrettyBox(halo, snap, nslice=10)
        plt.figure(1)
        ahalo.plot_halo_hist(color=colors[sim])
        plt.figure()
        print "sim:",sim,"snap: ",snap
        ahalo.plot_sigma_DLA()
        plt.ylim(1,1e5)
        plt.xlim(1e9,1e12)
        save_figure(path.join(outdir, "halos/cosmo"+str(sim)+"_sigmaDLA_z"+str(snap)))
        plt.clf()
    plt.figure(1)
    save_figure(path.join(outdir, "halos/cosmo_halohist_z"+str(snap)))
    plt.clf()

def get_rhohi_dndx(sim, ff=True, box=25):
    """Plot rho_HI and dndx across redshift"""
    halo = myname.get_name(sim, ff, box)
    snaps = {4:1, 3.5:2, 3:3, 2.5:4, 2:5}
    dndx=[]
    rhohi=[]
    zzz = []
    for zzzz in (4, 3.5, 3, 2.5, 2):
        try:
            ahalo = dp.PrettyBox(halo, snaps[zzzz], nslice=10)
            dndx.append(ahalo.line_density())
            rhohi.append(ahalo.omega_DLA())
            zzz.append(zzzz)
            del ahalo
        except IOError:
            continue
    return (zzz, dndx, rhohi)

def plot_rel_res(sim):
    """Load and make a plot of the difference between two simulations"""
    basel = myname.get_name(sim)
    bases = myname.get_name(sim, box=10)
#     plt.figure(1)
    for snap in (1, 3, 5):
        base = dp.PrettyBox(basel, snap, nslice=10)
        cddf_base = base.column_density_function()
        ahalo2 = dp.PrettyBox(bases, snap, nslice=10)
        cddf = ahalo2.column_density_function()
        plt.semilogx(cddf_base[0], np.log10(cddf[1]/cddf_base[1]), color=colors[snap], ls=lss[snap])
        if snap == 3:
            plt.figure(3)
            base.plot_column_density(color=colors[sim], ls=lss[sim], moment=True)
            ahalo2.plot_column_density(color="grey", ls=lss[sim], moment=True)
            dla_data.column_density_data(moment=True)
            save_figure(path.join(outdir,"cosmo_res_cddf_z3_abs"))
            plt.clf()
            plt.figure(1)
    savefile = path.join(basel,"snapdir_003","boxhi_grid_noH2.hdf5")
    base = dp.PrettyBox(basel, 3, nslice=10,savefile=savefile)
    cddf_base = base.column_density_function()
    savefile = path.join(bases,"snapdir_003","boxhi_grid_noH2.hdf5")
    ahalo2 = dp.PrettyBox(bases, 3, nslice=10,savefile=savefile)
    cddf = ahalo2.column_density_function()
    plt.semilogx(cddf_base[0], np.log10(cddf[1]/cddf_base[1]), color=colors[0], ls=lss[0])
    plt.ylim(-0.5,0.5)
    save_figure(path.join(outdir,"cosmo_res_cddf_z"+str(sim)))
    plt.clf()


def plot_rel_cddf(snap):
    """Load and make a plot of the difference between two simulations"""
    basen = myname.get_name(5)
    base = dp.PrettyBox(basen, snap, nslice=10)
    cddf_base = base.column_density_function()
    for xx in (0,1,2,3,4):
        halo2 = myname.get_name(xx)
        ahalo2 = dp.PrettyBox(halo2, snap, nslice=10)
        cddf = ahalo2.column_density_function()
        plt.semilogx(cddf_base[0], np.log10(cddf[1]/cddf_base[1]), color=colors[xx], ls=lss[xx])
    plt.ylim(-0.5,0.5)
    plt.xlim(1e17, 1e22)
    save_figure(path.join(outdir,"cosmo_rel_cddf_z"+str(snap)))
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
    plot_H2_effect(5,3)
    plot_rel_res(5)
    #plot_UVB_effect()
    plot_all_rho()
    plot_cutoff()
    #Make a plot of the column density functions.
    for ss in xrange(6):
        plot_cddf_a_halo(ss, 3)

    dla_data.column_density_data()

    save_figure(path.join(outdir,"cosmo_cddf_z3"))
    plt.clf()

    #Plot first moment
    for zz in (1,3,5):
        for ss in xrange(6):
            plot_cddf_a_halo(ss, zz, moment=True)

        if zz==3 :
            dla_data.column_density_data(moment=True)

        save_figure(path.join(outdir,"cosmo_cddf_z"+str(zz)+"_moment"))
        plt.clf()

    for zz in (1,3,5):
        plot_rel_cddf(zz)
#         plot_halohist(zz)

    #Make a plot of the effect of AGN on the cddf.
    for ss in (2,1):
        plot_cddf_a_halo(ss, 5, moment=True)
    for ss in (0,4):
        plot_cddf_a_halo(ss, 5, moment=True)
    #dla_data.column_density_data(moment=True)
    plt.xlim(1e17,3e22)
    plt.ylim(1e-4,0.3)
    save_figure(path.join(outdir,"cosmo_cddf_agn_z2"))
    plt.clf()

    #Make a plot of the effect of modifying the minimum velocity
    for ss in (0,1,5):
        plot_cddf_a_halo(ss, 3, moment=True)
    dla_data.column_density_data(moment=True)
    plt.xlim(1e17,3e22)
    plt.ylim(1e-4,0.3)
    save_figure(path.join(outdir,"cosmo_cddf_minvel_z3"))
    plt.clf()

    #Metallicity
    for zz in (1,3,5):
        zrange = {1:(7,3.5), 3:None, 5:(2.5,0)}
        for ss in xrange(6):
            plot_metal_halo(ss, zz)

        vel_data.plot_alpha_metal_data(zrange[zz])
        plt.xlim(-3,0)
        save_figure(path.join(outdir,"cosmo_metal_z"+str(zz)))
        plt.clf()

#     for zz in (1,3,5):
#         zrange = {1:(7,3.5), 3:None, 5:(2.5,0)}
#         for ss in xrange(6):
#             plot_metal_halo(ss, zz,lls=True)
#
#         vel_data.plot_alpha_metal_data(zrange[zz])
#
#         save_figure(path.join(outdir,"cosmo_lls_metal_z"+str(zz)))
#         plt.clf()
#
