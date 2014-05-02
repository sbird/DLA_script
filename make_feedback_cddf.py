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
import halocat
from dla_plots import tight_layout_wrapper
from save_figure import save_figure

outdir = myname.base + "plots/grid"

#Colors and linestyles for the simulations
colors = {0:"red", 1:"purple", 2:"cyan", 3:"green", 4:"darkslateblue", 5:"pink", 7:"blue", 6:"grey",8:"pink", 9:"orange",'A':'grey'}
lss = {0:"--",1:":", 2:":",3:"-.", 4:"--", 5:"--",6:"--",7:"-",8:"--", 9:"-",'A':"-"}
labels = {0:"DEF",1:"HVEL", 2:"HVNOAGN",3:"NOSN", 4:"WMNOAGN", 5:"MVEL",6:"METAL",7:"2xUV", 8:"FAST", 9:"FAST", 'A':"MOM"}
redshifts = {1:4, 2:3.5, 3:3, 4:2.5, 5:2}

def plot_cddf_a_halo(sim, snap, ff=True, moment=False):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10, label=labels[sim])
    ahalo.plot_column_density(color=colors[sim], ls=lss[sim], moment=moment)
    del ahalo

def plot_metal_halo(sim, snap, ff=True, lls=False):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10, label=labels[sim])
    if lls:
        ahalo.plot_lls_metallicity(color=colors[sim], ls=lss[sim])
    else:
        ahalo.plot_dla_metallicity(color=colors[sim], ls=lss[sim])
    del ahalo

def plot_mass_metal(sims, snap, ff=True):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10, label=labels[sim])
    ahalo.plot_dla_mass_metallicity(color=colors[sim])


def plot_metal_ion_corr(sim, snap,species="Si",ion=2, dla=True, othersave="boxhi_grid_H2_no_atten.hdf5"):
    """Plot metallicity from GFM_Metallicity vs from a single species for computing ionisation corrections"""
    halo = myname.get_name(sim)
    ahalo = dp.PrettyBox(halo, snap, nslice=10, label=labels[sim])
    ahalo.plot_dla_metallicity(color="red",ls="--")
    ahalo.plot_species_fraction(species, ion, dla, color="blue", ls="-")
    ahalo_no_atten = dp.PrettyBox(halo, snap, nslice=10, label=labels[sim],savefile=othersave)
    ahalo_no_atten.plot_species_fraction(species, ion, dla, color="green", ls="-.")
    vel_data.plot_alpha_metal_data((3.5,2.5))
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_ion_corr"+str(snap)))
    plt.clf()
    ahalo.plot_ion_corr(species, ion, dla,upper=1,lower=-1)
    ahalo_no_atten.plot_ion_corr(species, ion, dla,color="green",ls="--",upper=1,lower=-1)
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_rel_ion_corr"+str(snap)))
    plt.clf()
    del ahalo

def plot_H2_effect(sim, snap):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, True)
    savefile = "boxhi_grid_noH2.hdf5"
    ahalo = dp.PrettyBox(halo, snap, nslice=10, savefile=savefile, label=r"No $H_2$")
    ahalo.plot_column_density(color="blue", ls="--", moment=True)
    savefile = "boxhi_grid_H2.hdf5"
    ahalo2 = dp.PrettyBox(halo, snap, nslice=10, savefile=savefile, label=r"$H_2$")
    ahalo2.plot_column_density(color="red",moment=True)
#     savefile = path.join(halo,"snapdir_"+str(snap).rjust(3,'0'),"boxhi_grid_H2-old.hdf5")
#     ahalo2 = dp.PrettyBox(halo, snap, nslice=10, savefile=savefile)
#     ahalo2.plot_column_density(color="green",moment=True)
    dla_data.column_density_data(moment=True)
#     dla_data.noterdaeme_12_data(path.join(path.dirname(__file__),"../dla_data"), moment=True)
    plt.legend(loc=3)
    plt.xlim(1e20,2e22)
    plt.ylim(1e-5,0.1)
#     plt.title("CDDF for "+labels[sim]+" at z="+str(redshifts[snap]))
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_H2_"+str(snap)))
    plt.clf()
    cddf_base = ahalo.column_density_function()
    cddf = ahalo2.column_density_function()
    plt.semilogx(cddf_base[0], np.log10(cddf[1]/cddf_base[1]), color=colors[sim], ls=lss[sim])
    plt.ylim(-0.5,0.5)
    tight_layout_wrapper()
    ax = plt.gca()
    ylab = ax.set_ylabel(r"$N_\mathrm{HI} f(N)$")
    save_figure(path.join(outdir, "cosmo_rel"+str(sim)+"_H2_"+str(snap)))
    plt.clf()

def plot_UVB_effect():
    """Load a simulation and plot its cddf"""
    for i in (0,5,7):
        halo = myname.get_name(i, True)
        ahalo = dp.PrettyBox(halo, 3, nslice=10)
        ahalo.plot_column_density(color=colors[i], ls=lss[i],moment=True)
    dla_data.column_density_data(moment=True)
    save_figure(path.join(outdir, "cosmo_UVB_3"))
    plt.clf()

def plot_cutoff():
    """Plot effect with a cutoff self-shielding"""
    halo = myname.get_name(0, True)
    savefile = "boxhi_grid_cutoff_H2.hdf5"
    ahalo = dp.PrettyBox(halo, 3, nslice=10, savefile=savefile)
    cutoff0 = ahalo.column_density_function()
    ahalo = dp.PrettyBox(halo, 3, nslice=10)
    normal0 = ahalo.column_density_function()
    halo = myname.get_name(1, True)
    savefile = "boxhi_grid_cutoff_H2.hdf5"
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
    halo = myname.get_name(7, True)
    savefile = "boxhi_grid_cutoff_H2_32678.hdf5"
    ahalo = dp.PrettyBox(halo, 5, nslice=30, savefile=savefile)
    ahalo.plot_column_density(color="blue", ls="--", moment=True)
#     savefile = path.join(halo,"snapdir_003/boxhi_grid_16384.hdf5")
    ahalo2 = dp.PrettyBox(halo, 5, nslice=10)

    ahalo2.plot_column_density(color="red",moment=True, ls="-.")
    dla_data.column_density_data(moment=True)
    save_figure(path.join(outdir, "cosmo7_grid_5"))
    plt.clf()
    cdf1 = ahalo.column_density_function()
    cdf2 = ahalo2.column_density_function()
    plt.semilogx(cdf1[0], cdf1[1]/cdf2[1], color="red", ls="-")
    save_figure(path.join(outdir, "cosmo7_grid_5_rel"))
    plt.clf()


def plot_covering_frac(sim, snap, ff=True):
    """Load a simulation and plot its cddf"""
    halo = myname.get_name(sim, ff)
    ahalo = dp.PrettyBox(halo, snap, nslice=10)
    ahalo.plot_sigma_DLA()
    del ahalo
    save_figure(path.join(outdir, "cosmo"+str(sim)+"_covering_z"+str(snap)))
    plt.clf()

def plot_halohist(snap, dla=True):
    """Plot a histogram of nearby halos"""
    for sim in (0,4,2,1,3,7,9):   #xrange(8):
        halo = myname.get_name(sim, True)
        ahalo = dp.PrettyBox(halo, snap, nslice=10, label=labels[sim])
        plt.figure(1)
#         if sim != 2 and sim != 4:
        ahalo.plot_halo_hist(dla=dla,color=colors[sim], ls=lss[sim])
        if sim == 5:
            plt.figure(35)
            ahalo.plot_halo_hist(dla=dla,color=colors[sim], ls=lss[sim],plot_error=True)
            plt.figure(34)
            ahalo.plot_sigma_DLA()
        plt.figure()
        if dla:
            print "sim:",sim,"snap: ",snap
            ahalo.plot_sigma_DLA()
            plt.title(labels[sim]+" at z="+str(redshifts[snap]))
        else:
            print "sim:",sim,"snap: ",snap
            ahalo.plot_sigma_LLS()
        plt.ylim(1,1e5)
        plt.xlim(5e7,1e13)
        if dla:
            save_figure(path.join(outdir, "halos/cosmo"+str(sim)+"_sigmaDLA_z"+str(snap)))
        else:
            save_figure(path.join(outdir, "halos/cosmo"+str(sim)+"_sigmaLLS_z"+str(snap)))
        plt.clf()
    plt.figure(1)
    if snap == 5:
        plt.legend(loc=1,ncol=2)
    else:
        plt.legend(loc=1)
    if dla:
        plt.title("Redshift "+str(redshifts[snap]))
        plt.xlim(1e8,1e13)
        plt.ylim(0,1.3)
        save_figure(path.join(outdir, "halos/cosmo_halohist_z"+str(snap)))
    else:
        save_figure(path.join(outdir, "halos/cosmo_halohist_lls_z"+str(snap)))
    plt.clf()

def multi_halohist(snap):
    """Plot selected simulations against each other in sigma_DLA"""
    small = myname.get_name(5, True,box=10)
    big = myname.get_name(5, True,box=25)
    ahalo = dp.PrettyBox(small, snap, nslice=10, label=labels[5])
    bighalo = dp.PrettyBox(big, snap, nslice=10, label=labels[5])
    bighalo.plot_sigma_DLA()
    ahalo.plot_sigma_DLA(color="blue", color2="blue")
    plt.ylim(1,1e5)
    plt.xlim(5e7,1e12)
    save_figure(path.join(outdir, "halos/cosmo5_10_sigmaDLA_z"+str(snap)))
    plt.clf()
    ahalo.plot_halo_hist(color=colors[0], ls=lss[0],plot_error=True)
    bighalo.plot_halo_hist(color=colors[5], ls=lss[5],plot_error=True)
    plt.ylim(0,1)
    plt.xlim(1e8,3e12)
    save_figure(path.join(outdir, "halos/cosmo_10_halohist_z"+str(snap)))
    plt.clf()
    for pair in ((1,2), (0,5), (0,7)):
        small = myname.get_name(pair[0])
        big = myname.get_name(pair[1])
        ahalo = dp.PrettyBox(small, snap, nslice=10, label=labels[5])
        bighalo = dp.PrettyBox(big, snap, nslice=10, label=labels[5])
        bighalo.plot_sigma_DLA()
        ahalo.plot_sigma_DLA(color="blue", color2="blue")
        plt.ylim(1,1e5)
        plt.xlim(5e7,1e12)
        save_figure(path.join(outdir, "halos/cosmo"+str(pair[0])+str(pair[1])+"_sigmaDLA_z"+str(snap)))
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

def plot_omegahi_breakdown(sim):
    """Make the rho_HI plot with labels etc"""
    halo = myname.get_name(sim, True, 25)
    snaps = {4:1, 3.5:2, 3:3, 2.5:4, 2:5}
    fractions=[]
    zzz = []
    omegadla = []
    for zzzz in (4, 3.5, 3, 2.5, 2):
        ahalo = dp.PrettyBox(halo, snaps[zzzz], nslice=10)
        (massbins, fracs) = ahalo.get_omega_hi_mass_breakdown()
        fractions.append(fracs)
        zzz.append(zzzz)
        omegadla.append(ahalo.omega_DLA())
    fractions = np.array(fractions)
    for i in xrange(np.size(fractions[0,:])-2):
        plt.plot(zzz,fractions[:,i+1], 'o', color=colors[i], ls=lss[i], label=dp.pr_num(np.log10(massbins[i]))+" - "+dp.pr_num(np.log10(massbins[i+1])))
    plt.plot(zzz,fractions[:,-1], 'o', color=colors[6], ls=lss[6], label="Field")
    plt.plot(zzz,omegadla, 'o', color=colors[sim], ls=lss[sim], label="Total")
    plt.xlabel("z")
    plt.ylabel(r"$10^3 \Omega_\mathrm{DLA}$")
    dla_data.omegahi_not()
    plt.xlim(2,4)
    plt.ylim(0,2.3)
    plt.legend(loc=1, ncol=2)
    tight_layout_wrapper()
    save_figure(path.join(outdir,"cosmo_rhohi_break"+str(sim)))
    plt.clf()

def plot_dndx_breakdown(sim):
    """Make the rho_HI plot with labels etc"""
    halo = myname.get_name(sim, True, 25)
    snaps = {4:1, 3.5:2, 3:3, 2.5:4, 2:5}
    fractions=[]
    zzz = []
    omegadla = []
    for zzzz in (4, 3.5, 3, 2.5, 2):
        ahalo = dp.PrettyBox(halo, snaps[zzzz], nslice=10)
        (massbins, fracs) = ahalo.get_omega_hi_mass_breakdown(False)
        fractions.append(fracs)
        zzz.append(zzzz)
        omegadla.append(ahalo.line_density())
    fractions = np.array(fractions)
    for i in xrange(np.size(fractions[0,:])-2):
        plt.plot(zzz,fractions[:,i+1], color=colors[i], ls=lss[i], label=dp.pr_num(np.log10(massbins[i]))+" - "+dp.pr_num(np.log10(massbins[i+1])))
    plt.plot(zzz,fractions[:,-1], color=colors[6], ls=lss[6], label="Field")
    plt.plot(zzz,omegadla, color=colors[sim], ls=lss[sim], label="Total")
    plt.xlabel("z")
    plt.ylabel(r"$dN/dX$")
    dla_data.dndx_not()
    dla_data.dndx_pro()
    plt.xlim(2,4)
    plt.ylim(0,0.15)
    plt.legend(loc=1, ncol=2)
    tight_layout_wrapper()
    save_figure(path.join(outdir,"cosmo_dndx_break"+str(sim)))
    plt.clf()

def plot_rel_res(sim):
    """Load and make a plot of the difference between two simulations"""
    basel = myname.get_name(sim)
    bases = myname.get_name(sim, box=10)
    plt.figure(1)
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
            base.plot_halo_hist(Mmin=1e7,color=colors[sim])
            ahalo2.plot_halo_hist(Mmin=1e7,color="grey")
            plt.ylim(0,0.1)
            save_figure(path.join(outdir,"cosmo_res_halohist"))
            plt.clf()
            plt.figure(1)
    savefile = "boxhi_grid_noH2.hdf5"
    base = dp.PrettyBox(basel, 3, nslice=10,savefile=savefile)
    cddf_base = base.column_density_function()
    savefile = "boxhi_grid_noH2.hdf5"
    ahalo2 = dp.PrettyBox(bases, 3, nslice=10,savefile=savefile)
    cddf = ahalo2.column_density_function()
    plt.semilogx(cddf_base[0], np.log10(cddf[1]/cddf_base[1]), color=colors[0], ls=lss[0])
    plt.ylim(-0.5,0.5)
    save_figure(path.join(outdir,"cosmo_res_cddf_z"+str(sim)))
    plt.clf()


def plot_rel_cddf(snap):
    """Load and make a plot of the difference between two simulations"""
    basen = myname.get_name(7)
    base = dp.PrettyBox(basen, snap, nslice=10)
    cddf_base = base.column_density_function()
    for xx in (0,4,2,1,3,9):
        halo2 = myname.get_name(xx)
        ahalo2 = dp.PrettyBox(halo2, snap, nslice=10)
        cddf = ahalo2.column_density_function()
        plt.semilogx(cddf_base[0], cddf[1]/cddf_base[1], color=colors[xx], ls=lss[xx], label=labels[xx])
    plt.legend(loc=3, ncol=2)
    plt.xlabel(r"$N_\mathrm{HI}$ (cm$^{-2}$)")
    plt.ylabel(r"$f(N)/f_\mathrm{"+labels[7]+"}(N)$")
    plt.title("CDDF relative to "+labels[7]+" at z="+str(redshifts[snap]))
    plt.ylim(-0.2,1.8)
    plt.xlim(1e17, 1e22)
    tight_layout_wrapper()
    save_figure(path.join(outdir,"cosmo_rel_cddf_z"+str(snap)))
    plt.clf()

def plot_agn_rel_cddf(snap):
    """Load and make a plot of the difference between both simulations with and without AGN"""
    basen = myname.get_name(0)
    base = dp.PrettyBox(basen, snap, nslice=10)
    cddf_base = base.column_density_function()
    basen = myname.get_name(4)
    other = dp.PrettyBox(basen, snap, nslice=10)
    cddf = other.column_density_function()
    plt.semilogx(cddf_base[0], cddf[1]/cddf_base[1], color=colors[0], ls=lss[0])
    basen = myname.get_name(1)
    base = dp.PrettyBox(basen, snap, nslice=10)
    cddf_base = base.column_density_function()
    basen = myname.get_name(2)
    other = dp.PrettyBox(basen, snap, nslice=10)
    cddf = other.column_density_function()
    plt.semilogx(cddf_base[0], cddf[1]/cddf_base[1], color=colors[1], ls=lss[1])
    plt.ylim(0.8,1.5)
    plt.xlim(1e17, 1e22)
    save_figure(path.join(outdir,"cosmo_rel_agn_cddf_z"+str(snap)))
    plt.clf()

def plot_halos(sim,hh):
    """Plot the halo closest in mass and position in the given sim to the halo of the given number in sim 7."""
    ahalo = dp.PrettyHalo(myname.get_name(sim),3,20000)
    ahalo7 = dp.PrettyHalo(myname.get_name(7),3,20000)
    (mass, cofm, radii) = _load_halo(ahalo, 30)
    (mass7, cofm7, _) = _load_halo(ahalo7, 30)
    dist = np.sum((cofm7[hh] - cofm)**2, axis=1)
    nn = np.where( dist == np.min(dist))
    rhh = hh
    hh = np.ravel(nn)[0]
    print "Requested: ",rhh,"Got: ",hh," dist:",np.sqrt(dist[nn])," r-mass:",mass[hh]/mass7[rhh], "ormass: ",mass7[rhh]
    plt.title(r"Central Halo: $"+dp.pr_num(ahalo.sub_mass[hh]/0.72/1e11)+r"\times 10^{11} M_\odot$")
    ahalo.plot_pretty_halo(hh)
    #Backwards because someone is a fortran programmer
    circle=plt.Circle((0,0),radii[hh],color="black",fill=False)
    ax = plt.gca()
    ax.add_artist(circle)
#     plot_rvir(ahalo.sub_cofm[hh], cofm, radii,ahalo.sub_radii[hh])
    dp.tight_layout_wrapper()
    save_figure(path.join(outdir,"pretty_"+str(sim)+"_halo_"+str(rhh)))
    plt.clf()
    ahalo.plot_pretty_cut_halo(hh)
    circle=plt.Circle((0,0),radii[hh],color="black",fill=False)
    ax = plt.gca()
    ax.add_artist(circle)
#     plot_rvir(ahalo.sub_cofm[hh], cofm, radii,ahalo.sub_radii[hh]/)
    plt.title(labels[sim]+" at z=3")
    dp.tight_layout_wrapper()
    save_figure(path.join(outdir,"pretty_cut_"+str(sim)+"_halo_"+str(rhh)))
    plt.clf()
    del ahalo

def plot_rvir(apos, cofm, radii, maxdist):
    """Plot black circles for virial radius"""
    zz = cofm[:,0]-apos[0]
    #Make this a bit bigger so we catch halos just slightly within our radius
    zcut = np.where(np.abs(zz) < 1.1*maxdist)
    zcofm = cofm[zcut,1:][0]
    #rel_r = np.sqrt(np.sum((zcofm - apos[1:])**2,1))
    #rr = np.where(rel_r < 400)
    dist = np.abs(zcofm-apos[1:])
    rr = np.where(np.logical_and(dist[:,0]< 2.*maxdist, dist[:,1] < 2.*maxdist))
    for r in rr[0]:
      pos = zcofm[r]-apos[1:]
      #Backwards because someone is a fortran programmer
      circle=plt.Circle((pos[1],pos[0]),radii[zcut][r],color="black",fill=False)
      ax = plt.gca()
      ax.add_artist(circle)

def _load_halo(self, minpart=400):
    """Load a halo catalogue"""
    #This is rho_c in units of h^-1 M_sun (kpc/h)^-3
    rhom = 2.78e+11* self.omegam / (1e3**3)
    #Mass of an SPH particle, in units of M_sun, x omega_m/ omega_b.
    target_mass = self.box**3 * rhom / self.npart[0]
    min_mass = target_mass * minpart / 1e10
    print "mass: ",min_mass
    (_, halo_mass, halo_cofm, halo_radii) = halocat.find_all_halos(self.snapnum, self.snap_dir, min_mass)
    return (halo_mass, halo_cofm, halo_radii)


def plot_all_rho(simlist):
    """Make the rho_HI plot with labels etc"""
    #Cosmo0
    for i in simlist:   #xrange(8):
        (zzz,dndx,omegadla) = get_rhohi_dndx(i)
        plt.figure(1)
        plt.plot(zzz,dndx, 'o', color=colors[i], ls=lss[i], label=labels[i])
        plt.figure(2)
        plt.plot(zzz,omegadla, 'o', color=colors[i], ls=lss[i],label=labels[i])
    plt.figure(1)
    plt.xlabel("z")
    plt.ylabel(r"$dN / dX$")
    dla_data.dndx_not()
    dla_data.dndx_pro()
    plt.xlim(2,4)
    plt.ylim(0,0.13)
    plt.legend(loc=4, ncol=3)
    tight_layout_wrapper()
    save_figure(path.join(outdir,"cosmo_dndx"))
    plt.clf()

    plt.figure(2)
    plt.xlabel("z")
    plt.ylabel(r"$10^3 \Omega_\mathrm{DLA}$")
    dla_data.omegahi_not()
#     dla_data.omegahi_pro()
    plt.xlim(2,4)
    plt.ylim(0,2.)
    plt.legend(loc=4, ncol=3)
    tight_layout_wrapper()
    save_figure(path.join(outdir,"cosmo_rhohi"))
    plt.clf()

def plot_breakdown(simlist):
    """Make a plot of the column density function, broken down by halo mass."""
    for sss in simlist:
        halo = myname.get_name(sss, True)
        for nn in (1,3,5):
            ahalo = dp.PrettyBox(halo, nn, nslice=10)
            ahalo.plot_colden_mass_breakdown()
            plt.xlim(20.3,22.3)
            plt.ylim(0,1.4)
            plt.legend(loc=2,ncol=2)
            save_figure(path.join(outdir,"halos/break/cosmo"+str(sss)+"_"+str(nn)+"_break"))
            plt.clf()
            del ahalo


if __name__ == "__main__":
    plot_H2_effect(7,3)
    simlist = (0,1,3,7,9)

    for sim in simlist:
        for halos in (15,17, 18,20,35,45,50):
            plot_halos(sim, halos)
    zrange = {1:(7,3.5), 3:(3.5,2.5), 5:(2.5,1.5)}
#     halo = myname.get_name(0)
#     ahalo = dp.PrettyBox(halo, 3, nslice=10, label=labels[0])
#     ahalo.plot_dla_metallicity(color="blue", ls="-")
#     vel_data.plot_alpha_metal_data(zrange[3])
#     plt.ylim(0,1)
#     plt.xlim(-3,0)
#     save_figure(path.join(outdir,"cosmo_metal_z3_lone"))
#     plt.clf()

    plot_breakdown(simlist)
    plot_omegahi_breakdown(7)
    plot_dndx_breakdown(7)
    plot_metal_ion_corr(0,3)
    plot_cddf_a_halo(7, 3)

#     dla_data.column_density_data()
#     ax = plt.gca()
#     ylab = ax.set_ylabel(r"$f(N)$")
#     save_figure(path.join(outdir,"cosmo_cddf_lone"))
#     plt.clf()

#     for ss in (1,3,5):
#         for sim in (1,5,6,7):
#             plot_mass_metal(sim,ss)
#         save_figure(path.join(outdir, "cosmo_mass_metal"+str(ss)))
#         plt.clf()
#     for ss in (1,3,5):
#         plot_mass_metal(7,ss)
#     save_figure(path.join(outdir, "cosmo_7mass_metal"))
#     plt.clf()
#     plot_mass_metal(7,3)
#     save_figure(path.join(outdir, "cosmo_7mass_metal3"))
#     plt.clf()
#     plot_rel_res(5)
    plot_grid_res()
#     plot_UVB_effect()
    plot_all_rho(simlist)
#     plot_cutoff()
    #Make a plot of the column density functions.
    for ss in simlist:   #xrange(6):
        plot_cddf_a_halo(ss, 3)

    plt.legend(loc=3)
    dla_data.column_density_data()

    ax = plt.gca()
    ylab = ax.set_ylabel(r"$f(N)$")
#     tight_layout_wrapper()
    save_figure(path.join(outdir,"cosmo_cddf_z3"))
    plt.clf()

    #Plot first moment
    for zz in (1,3,4,5):
        for ss in simlist:   #xrange(6):
            if zz == 4 and ss == 3:
                plot_cddf_a_halo(3, 3, moment=True)
            else:
                plot_cddf_a_halo(ss, zz,moment=True)

        if zz==3 :
            dla_data.column_density_data(moment=True)
        if zz == 4:
            dla_data.noterdaeme_12_data(path.join(path.dirname(__file__),"../dla_data"), moment=True)
            plt.xlim(1e20,5e22)
            plt.ylim(ymax=0.1)
#             dla_data.zafar_data(path.join(path.dirname(__file__),"../dla_data"), moment=True)

        plt.legend(loc=3)
        ax = plt.gca()
        ylab = ax.set_ylabel(r"$N_\mathrm{HI} f(N)$")
        tight_layout_wrapper()
        save_figure(path.join(outdir,"cosmo_cddf_z"+str(zz)+"_moment"))
        plt.clf()

    for zz in (1,3,5):
        plot_rel_cddf(zz)
        plot_halohist(zz)
#         multi_halohist(zz)
#         plot_halohist(zz, False)

    #Make a plot of the effect of AGN on the cddf.
    for ss in (2,1):
        plot_cddf_a_halo(ss, 4, moment=True)
    for ss in (0,4):
        plot_cddf_a_halo(ss, 4, moment=True)
    plt.legend(loc=3)
    dla_data.column_density_data(moment=True)
    plt.xlim(1e17,3e22)
    plt.ylim(1e-4,0.3)
    save_figure(path.join(outdir,"cosmo_cddf_agn_z2"))
    plt.clf()
    plot_agn_rel_cddf(5)

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
        zrange = {1:(7,3.5), 3:(3.5,2.5), 5:(2.5,1.5)}
        for ss in simlist:   #xrange(6):
            plot_metal_halo(ss, zz)

        vel_data.plot_alpha_metal_data(zrange[zz])
        plt.legend(loc=2)
        plt.ylim(0,1.45)
        plt.xlim(-3,0)
        plt.title("DLA metallicity at z="+str(redshifts[zz]))
        save_figure(path.join(outdir,"cosmo_metal_z"+str(zz)))
        plt.clf()


    #LLS Metallicity
    for zz in (1,3,5):
        for ss in simlist+(4,6):
            plot_metal_halo(ss, zz, lls=True)
        plt.legend(loc=2, ncol=3)
        plt.ylim(0,1.2)
        save_figure(path.join(outdir,"cosmo_lls_metal_z"+str(zz)))
        plt.clf()


#     #At z=0.55
#     plot_metal_halo(0, 101,lls=True)
#     vel_data.plot_lls_metal_data()
#
#     save_figure(path.join(outdir,"cosmo_lls_metal_z101"))
#     plt.clf()
#
