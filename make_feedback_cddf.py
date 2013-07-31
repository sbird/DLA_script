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
from save_figure import save_figure

outdir = myname.base + "plots/grid"

#Colors and linestyles for the simulations
colors = {0:"red", 1:"purple", 2:"blue", 3:"green", 4:"orange"}
lss = {0:"--",1:":", 2:"-",3:"-.", 4:"-"}

def plot_cddf_a_halo(sim, snap, ff=True, moment=False):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10)
    ahalo.plot_column_density(color=colors[sim], ls=lss[sim], moment=moment)
    del ahalo

def plot_H2_effect(sim, snap):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
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

def plot_covering_frac(sim, snap, ff=True):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10)
    ahalo.plot_sigma_DLA()
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_covering_z"+str(snap)))
    plt.clf()
    ahalo.plot_halo_hist()
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_halohist_z"+str(snap)))
    del ahalo
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

def plot_rel_cddf(sim1, sim2, snap, color="black"):
    """Load and make a plot of the difference between two simulations"""
    halo1 = myname.get_name(sim1, ff)
    halo2 = myname.get_name(sim2, ff)
    ahalo1 = dp.PrettyBox(halo1, snap, nslice=10)
    cddf1 = ahalo1.column_density_function(0.2,16,24)
    del ahalo1
    ahalo2 = dp.PrettyBox(halo2, snap, nslice=10)
    cddf2 = ahalo2.column_density_function(0.2,16,24)
    plt.semilogx(cddf1[0], cddf2[1]/cddf1[1], color=color)

def plot_all_rho():
    """Make the rho_HI plot with labels etc"""
    #Cosmo0
    zzz={}
    dndx={}
    omega={}
    zzz[0]= [4, 3, 2]
    dndx[0]= [0.12982373590447616, 0.11318656438441635, 0.14167778407440484]
    omega[0]= [1.1295349304295403, 1.0832253887624965, 1.4772732823119952]
    #Cosmo1
    zzz[1]= [4, 3, 2]
    dndx[1]= [0.076274082906110152, 0.050977933084835461, 0.043776291837515323]
    omega[1]= [0.68763319854249694, 0.49411308856267733, 0.42703152981413667]
    #Cosmo2
    zzz[2]= [4, 3.5, 3, 2.5, 2]
    dndx[2]= [0.081675998741546157, 0.071281732587967661, 0.061506461643187245, 0.061143942748744885, 0.063275568]
    omega[2]= [0.76470263480711975, 0.69383248829835542, 0.64764348287042328, 0.6917879571651826, 0.7428807770637]
    #Cosmo3
    zzz[3]= [4, 3, 2]
    dndx[3]= [0.11528106522481622, 0.11797023730537365, 0.12545022290952848]
    omega[3]= [1.7090510472291021, 1.8510973567379598, 1.9917860682826971]
    #Cosmo4
    zzz[4]= [4, 3, 2]
    dndx[4]= [0.12485554993298456, 0.11632137558129531, 0.14296572069204525]
    omega[4]= [1.1592268064398157, 1.2041971815381218, 1.6341245991834694]
#     for i in (0,1,2,3,4):
#         (zzz,dndx,rhohi) = get_rhohi_dndx(i)
#         plt.figure(1)
#         print "zzz=",zzz
#         print "dndx=",dndx
#         print "omega=",rhohi
#         plt.plot(zzz,dndx, color=colors[i], ls=lss[i])
#         plt.figure(2)
#         plt.plot(zzz,rhohi, color=colors[i], ls=lss[i])
#
    plt.figure(1)
    for ss in xrange(0,5):
        plt.plot(zzz[ss],dndx[ss], color=colors[ss], ls=lss[ss])
    plt.xlabel("z")
    plt.ylabel(r"$dN / dX$")
    dla_data.dndx_not()
    dla_data.dndx_pro()
    save_figure(path.join(outdir,"cosmo_dndx"))
    plt.clf()

    plt.figure(2)
    for ss in xrange(0,5):
        plt.plot(zzz[ss],omega[ss], color=colors[ss], ls=lss[ss])
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
    plot_all_rho()
    #Make a plot of the column density functions.
#     for ss in (3,2,0):
#         plot_cddf_a_halo(ss, 3)
#
#     dla_data.column_density_data()
#
#     save_figure(path.join(outdir,"cosmo_cddf_z3"))
#     plt.clf()

    #Plot first moment
    for ss in (4,3,2,1,0):
        plot_cddf_a_halo(ss, 3, moment=True)

    dla_data.column_density_data(moment=True)

    save_figure(path.join(outdir,"cosmo_cddf_z3_moment"))
    plt.clf()

    plot_breakdown()
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
    # for i in (0,2,3):
    #     plot_covering_frac(i,3, True)

    #Make a plot of the effect of AGN on the cddf.
    for ss in (2,1):
        plot_cddf_a_halo(ss, 3, moment=True)
    dla_data.column_density_data()

    save_figure(path.join(outdir,"cosmo_cddf_agn_z3"))
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
