# -*- coding: utf-8 -*-
"""Script for making various DLA-related plots"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt
import os.path as path
import dla_plots as dp
import dla_data
import numpy as np
from save_figure import save_figure
import sys

#Argument: 2 => pretty density plots
#          3 => sigma_DLA plots
#          4 => Column density plots
#          5 => dNdz and halo function
#          6 => Model fits
bases=[
# "/home/spb/data/ComparisonProject/128_20Mpc",
# "/home/spb/data/ComparisonProject/256_20Mpc",
"/home/spb/data/ComparisonProject/512_20Mpc",
]
minpart=400
snaps=[
# 31,
# 41,
90,
# 124,
141,
191,
# 314,
]

nosmall=0

def print_stuff(string):
    """Routine to print the parameters of the fit in a format that can be inserted into python"""
    for (bs,snp) in [(b,s) for b in bases for s in snaps]:
        adir=path.join(bs,string)
        ahalo=dp.PrettyHalo(adir,snp,minpart)
        ap=ahalo.get_sDLA_fit()
        print ahalo.snapnum," : [",
        for a in ap:
            print a,',',
        print '],'
        del ahalo

if len(sys.argv) > 1 and int(sys.argv[1]) == 6:
    #This line is here because otherwise MKL
    #dlloads things in the wrong order and segfaults.
    plt.figure()
    print " arepo_halo_p = {"
    print_stuff("Arepo_ENERGY")
    print '}'
    print " gadget_halo_p = {"
    print_stuff("Gadget")
    print '}'
    sys.exit()

#Plots with the nearest 200kpc
for (base,snapnum) in [(bb,ss) for bb in bases for ss in snaps]:
    outdir=path.join(base,"plots")
    print "Saving plots for snapshot ",snapnum," to ",outdir

    plt.figure(1)

    #Load only the nHI grids
    hplots=dp.HaloHIPlots(base,snapnum,minpart=minpart)

    if len(sys.argv) < 2 or int(sys.argv[1]) == 2:
        #low-mass halo radial profile
        hplots.plot_radial_profile(minM=3e9, maxM=3.5e9,maxR=10.)
        plt.ylim(ymax = 27)
        save_figure(path.join(outdir,"radial_profile_halo_low_"+str(snapnum)))
        plt.clf()

        hplots.ahalo.plot_pretty_halo(0)
        plt.title(r"Arepo, $\mathrm{z}="+dp.pr_num(hplots.ahalo.redshift,1)+"$")
        dp.tight_layout_wrapper()
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_halo"))
        plt.clf()
        #Get the right halo
        g_halo_0 = hplots.ghalo.identify_eq_halo(hplots.ahalo.sub_mass[0],hplots.ahalo.sub_cofm[0,:],maxpos=50.)[0]
        try:
            a_shalo=np.min(np.where(hplots.ahalo.sub_mass < 1e10))
            #Get the right halo for a smaller halo
            s_mass=hplots.ahalo.sub_mass[a_shalo]
            s_pos=hplots.ahalo.sub_cofm[a_shalo,:]
            g_shalo = hplots.ghalo.identify_eq_halo(s_mass,s_pos)
            #Make sure it has a gadget counterpart
            if np.size(g_shalo) == 0:
                nosmall=1
                for i in np.where(hplots.ahalo.sub_mass < 1e10)[0]:
                    s_mass=hplots.ahalo.sub_mass[i]
                    s_pos=hplots.ahalo.sub_cofm[i,:]
                    g_shalo = hplots.ghalo.identify_eq_halo(s_mass,s_pos)
                    if np.size(g_shalo) != 0 :
                        a_shalo=i
                        nosmall=0
                        break
        except ValueError:
            nosmall=1

        hplots.ghalo.plot_pretty_halo(g_halo_0)
        plt.title(r"Gadget, $\mathrm{z}="+dp.pr_num(hplots.ahalo.redshift,1)+"$")
        dp.tight_layout_wrapper()
        save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"pretty_halo"))

        plt.clf()
        hplots.ahalo.plot_pretty_cut_halo(0)
        plt.title(r"Arepo, $\mathrm{z}="+dp.pr_num(hplots.ahalo.redshift,1)+"$")
        dp.tight_layout_wrapper()
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_cut_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_cut_halo(g_halo_0)
        plt.title(r"Gadget, $\mathrm{z}="+dp.pr_num(hplots.ahalo.redshift,1)+"$")
        dp.tight_layout_wrapper()
        save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"pretty_cut_halo"))

        plt.clf()
        if not nosmall:
            hplots.ahalo.plot_pretty_halo(a_shalo)
            save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"_small_pretty_halo"))
            plt.clf()
            hplots.ghalo.plot_pretty_halo(g_shalo[0])
            save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"_small_pretty_halo"))

            plt.clf()
            hplots.ahalo.plot_pretty_cut_halo(a_shalo)
            save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"_small_pretty_cut_halo"))
            plt.clf()
            hplots.ghalo.plot_pretty_cut_halo(g_shalo[0])
            save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"_small_pretty_cut_halo"))

            plt.clf()

    if len(sys.argv) < 2 or int(sys.argv[1]) == 3:
        plt.clf()
        hplots.plot_sigma_DLA()
        save_figure(path.join(outdir,"sigma_DLA_"+str(snapnum)))

        plt.clf()
        hplots.plot_sigma_DLA(17,20.3)
        save_figure(path.join(outdir,"sigma_DLA_17_"+str(snapnum)))

    if len(sys.argv) < 2 or int(sys.argv[1]) == 4:
        #Fig 12
        plt.clf()
        hplots.plot_column_density()
        if snapnum == 141:
            dla_data.column_density_data()
        save_figure(path.join(outdir,"columden_"+str(snapnum)))

        plt.clf()
        hplots.plot_column_density_breakdown()
        save_figure(path.join(outdir,"columden_break_"+str(snapnum)))

        #Fig 12
#         plt.clf()
#         hplots.plot_rel_column_density()
#         plt.ylim(0.5,1.5)
#         save_figure(path.join(outdir,"columden_rel_"+str(snapnum)))

    if len(sys.argv) < 2 or int(sys.argv[1]) == 5:
        #Fig 11
        plt.clf()
        hplots.plot_dN_dla()
        save_figure(path.join(outdir,"dNdz_"+str(snapnum)))

        #Fig 5
        plt.clf()
        hplots.plot_halo_mass_func()
        save_figure(path.join(outdir,"halo_func_"+str(snapnum)))
        plt.clf()

    del hplots
