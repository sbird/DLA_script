"""Script for making various DLA-related plots"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt
import os.path as path
import dla_plots as dp
from save_figure import *

bases=[
"/home/spb/data/ComparisonProject/128_20Mpc",
"/home/spb/data/ComparisonProject/256_20Mpc",
"/home/spb/data/ComparisonProject/512_20Mpc",
]
minpart=1000
snaps=[
91,
124,
191,
]

for (base,snapnum) in [(bb,ss) for bb in bases for ss in snaps]:
    outdir=path.join(base,"plots")
    print "Saving plots for snapshot ",snapnum," to ",outdir

    #Load only the gas grids
    hplots=dp.HaloHIPlots(base,snapnum,minpart=minpart,skip_grid=1)
    plt.figure()
    hplots.ahalo.plot_pretty_gas_halo()
    save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_gas_halo"))
    plt.clf()
    hplots.ghalo.plot_pretty_gas_halo()
    save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"pretty_gas_halo"))
    #Radial profiles
    plt.clf()
    hplots.plot_radial_profile(0)

    del hplots
    #Load only the nHI grids
    hplots=dp.HaloHIPlots(base,snapnum,minpart=minpart,skip_grid=2)
    hplots.plot_radial_profile(0)
    save_figure(path.join(outdir,"radial_profile_halo_0_"+str(snapnum)))


    #Fig 6
    plt.clf()
    hplots.ahalo.plot_pretty_halo()
    save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_halo"))
    plt.clf()
    hplots.ghalo.plot_pretty_halo()
    save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"pretty_halo"))

    #Fig 9
    plt.clf()
    dp.plot_totalHI(base,snapnum)
    save_figure(path.join(outdir,"total_HI_"+str(snapnum)))

    #Fig 10
    plt.clf()
    hplots.plot_sigma_DLA()
    save_figure(path.join(outdir,"sigma_DLA_"+str(snapnum)))

    #Fig 10
    plt.clf()
    hplots.plot_sigma_DLA(17)
    save_figure(path.join(outdir,"sigma_DLA_17_"+str(snapnum)))

    #Fig 10
    plt.clf()
    hplots.plot_rel_sigma_DLA()
    save_figure(path.join(outdir,"rel_sigma_DLA_"+str(snapnum)))

#    #Fig 11
#    plt.clf()
#    hplots.plot_dN_dla()
#    save_figure(path.join(outdir,"dNdz_"+str(snapnum)))

    #Fig 12
    plt.clf()
    hplots.plot_column_density()
    save_figure(path.join(outdir,"columden_"+str(snapnum)))

    #Fig 12
    plt.clf()
    hplots.plot_rel_column_density()
    save_figure(path.join(outdir,"columden_rel_"+str(snapnum)))

#     #Fig 5
#     plt.clf()
#     hplots.plot_halo_mass_func()
#     save_figure(path.join(outdir,"halo_func_"+str(snapnum)))
