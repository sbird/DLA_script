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

#Argument: 1 => totalHI plots
#          2 => pretty density plots
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
#Plots with all the halo particles
if len(sys.argv) > 1 and int(sys.argv[1]) == 1:
    for (base,snapnum) in [(bb,ss) for bb in bases for ss in snaps]:
        outdir=path.join(base,"plots")
        print "Saving total plots for snapshot ",snapnum," to ",outdir
        #Fig 9
        tot=dp.TotalHIPlots(base,snapnum,minpart)
#         plt.figure()
#         tot.plot_totalHI()
#         save_figure(path.join(outdir,"total_HI_"+str(snapnum)))

#         plt.clf()
#         tot.plot_MHI()
#         save_figure(path.join(outdir,"MHI_vs_Mgas"+str(snapnum)))
#         plt.clf()
#
        tot.plot_gas()
        save_figure(path.join(outdir,"halo_vs_gas_"+str(snapnum)))
        plt.clf()

def print_stuff(string):
    for (base,snapnum) in [(bb,ss) for bb in bases for ss in snaps]:
        adir=path.join(base,string)
        ahalo=dp.PrettyHalo(adir,snapnum,minpart,skip_grid=2)
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

    plt.figure()

    if  len(sys.argv) < 2 or int(sys.argv[1]) == 2:
        #Load only the gas grids
        hplots=dp.HaloHIPlots(base,snapnum,minpart=minpart,skip_grid=1)
        #Find a smallish halo
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
        #Get the right halo
        g_halo_0 = hplots.ghalo.identify_eq_halo(hplots.ahalo.sub_mass[0],hplots.ahalo.sub_cofm[0,:],maxpos=50.)[0]


        hplots.ahalo.plot_pretty_gas_halo(0)
        save_figure(path.join(outdir,"arepo_"+str(snapnum)+"pretty_gas_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_gas_halo(g_halo_0)
        save_figure(path.join(outdir,"gadget_"+str(snapnum)+"pretty_gas_halo"))
        plt.clf()
        if not nosmall:
            hplots.ahalo.plot_pretty_gas_halo(a_shalo)
            save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"_small_pretty_gas_halo"))
            plt.clf()
            hplots.ghalo.plot_pretty_gas_halo(g_shalo[0])
            save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"_small_pretty_gas_halo"))
            plt.clf()
#             hplots.ahalo.plot_pretty_cut_gas_halo(a_shalo)
#             save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"_small_pretty_cut_gas_halo"))
#             plt.clf()
#             hplots.ghalo.plot_pretty_cut_gas_halo(g_shalo[0])
#             save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"_small_pretty_cut_gas_halo"))
#             plt.clf()
        hplots.ahalo.plot_pretty_cut_gas_halo(0)
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"_pretty_cut_gas_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_cut_gas_halo(g_halo_0)
        save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"_pretty_cut_gas_halo"))

        #Radial profiles
        plt.clf()
        del hplots

    #Load only the nHI grids
    hplots=dp.HaloHIPlots(base,snapnum,minpart=minpart,skip_grid=2)

    if len(sys.argv) < 2 or int(sys.argv[1]) == 2:
        #low-mass halo radial profile
        hplots.plot_radial_profile(minM=2.5e9, maxM=3e9,maxR=10.)
        save_figure(path.join(outdir,"radial_profile_halo_low_"+str(snapnum)))
        plt.clf()

#         plt.figure(1)
#         hplots.plot_radial_profile(minM = 1e11,maxM=1.5e11,maxR=100.)
#         save_figure(path.join(outdir,"radial_profile_halo_0_"+str(snapnum)))
        #Fig 6
        plt.clf()
        hplots.ahalo.plot_pretty_halo(0)
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_halo(g_halo_0)
        save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"pretty_halo"))

        plt.clf()
        hplots.ahalo.plot_pretty_cut_halo(0)
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_cut_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_cut_halo(g_halo_0)
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

        print "Arepo: Omega_DLA=",hplots.ahalo.omega_DLA(21.75)
        print "Gadget: Omega_DLA=",hplots.ghalo.omega_DLA(21.75)

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

def plot_omega_DLA():
    #Cutoff is NHI = 21.75
    ar_om_DLA = [0.0438255320185,0.0877819801446,0.713437467697,1.46144716284,2.60101048655]
    gad_om_DLA = [0.084774233630, 0.180307848083,0.36464657397931499,0.70912333103719749,1.22997188625,1.95213972877,2.63684502775]
    zz = [8,7,6,5,4,3,2]
    zza = [8,7,4,3,2]
    plt.plot(zza,ar_om_DLA,'^',color="blue",ls="-")
    plt.plot(zz,gad_om_DLA,'s',color="red",ls="--")
    plt.ylabel(r"$1000 \Omega_\mathrm{DLA}$")
    plt.xlabel(r"z")

