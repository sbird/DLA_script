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
outdir = base + "plots/"

def plot_cddf_a_halo(sim, snap, color="red"):
   """Load a simulation and plot its cddf"""
   halo = "Cosmo"+str(sim)+"_V6"
   ahalo = dp.PrettyBox(base+halo, snap, nslice=10)
   ahalo.plot_column_density(color=color)
   del ahalo

colors=["red", "blue", "orange", "purple"]

#Make a plot of the column density functions.
for ss in (3,2,1,0):
   plot_cddf_a_halo(ss, 60, color=colors[ss])

dla_data.column_density_data()

save_figure(path.join(outdir,"cosmo_cddf_z3"))
plt.clf()

#A plot of the redshift evolution
zz = [54,60,68]
for ii in (0,1,2):
   plot_cddf_a_halo(0, zz[ii], color=colors[ii])

dla_data.column_density_data()
plt.title("Column density function at z=4-2")
save_figure(path.join(outdir,"cosmo0_cddf_zz"))
plt.clf()


def plot_rel_cddf(sim1, sim2, snap, color="black"):
   """Load and make a plot of the difference between two simulations"""
   halo1 = "Cosmo"+str(sim1)+"_V6"
   halo2 = "Cosmo"+str(sim2)+"_V6"
   ahalo1 = dp.PrettyBox(base+halo1, snap, nslice=10)
   cddf1 = ahalo1.column_density_function(0.2,16,24)
   del ahalo1
   ahalo2 = dp.PrettyBox(base+halo2, snap, nslice=10)
   cddf2 = ahalo2.column_density_function(0.2,16,24)
   plt.semilogx(cddf1[0], cddf2[1]/cddf1[1], color=color)

plot_rel_cddf(1,0,60)
plot_rel_cddf(2,0,60, color="grey")

save_figure(path.join(outdir,"cosmo_rel_cddf_z3"))
plt.clf()

def plot_cddf_breakdown(sim, snap, color="red"):
     """Load a simulation and plot its cddf"""
     halo = "Cosmo"+str(sim)+"_V6"
     ahalo = dp.PrettyHalo(base+halo, snap)
     ahalo.plot_column_density_breakdown(color=color)
     del ahalo

plot_cddf_breakdown(0, 60)

plot_cddf_breakdown(3, 60, color="blue")
plt.xlim(5e20,1e23)
plt.ylim(5e-2,2)
save_figure(path.join(outdir,"cosmo0_cddf_break_z3"))
plt.clf()

def plot_halo(sim, snap, num):
    """Load a simulation and plot its cddf"""
    halo = "Cosmo"+str(sim)+"_V6"
    ahalo = dp.PrettyHalo(base+halo, snap)
    ahalo.plot_pretty_halo(num)

def plot_metal_halo(sim, snap, num):
    """Load a simulation and plot its cddf"""
    halo = "Cosmo"+str(sim)+"_V6"
    ahalo = dp.PrettyMetal(base+halo, snap, "Si", 2)
    ahalo.plot_pretty_halo(num)

plot_metal_halo(0,60,15)
plt.xlim(-305,305)
plt.ylim(-305, 305)
save_figure(path.join(outdir,"cosmo0_metal_halo_z3"))
plt.clf()

plot_halo(0,60,15)
plt.xlim(-305,305)
plt.ylim(-305, 305)
save_figure(path.join(outdir,"cosmo0_halo_z3"))
plt.clf()

