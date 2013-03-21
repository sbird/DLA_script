# -*- coding: utf-8 -*-
"""Script for making Lyman-alpha forest temperature-density plots using the phase_plot module"""

import matplotlib
matplotlib.use('PDF')

import phase_plot
import matplotlib.pyplot as plt
outdir="/home/spb/scratch/finals/"
bar_label="Mass ($10^{6} M_\odot$ h$^{-1}$)"
ar=phase_plot.get_mass_map(13,"/home/spb/data/finals/FINAL_E_NC/output")

fig=plt.figure()

plt.imshow(ar.map,origin='lower',extent=ar.get_lims(),aspect='auto',vmin=0,vmax=100,cmap="gist_heat_r")
plt.colorbar()
plt.xlabel(r"$\log_{10}\, Z/ Z_sun$",size=15)
plt.ylabel(r"$\log_{10} \left(\rho /\rho_\mathrm{c}\right)$",size=15)
# plt.tight_layout()
plt.savefig(outdir+"met_phase.pdf")


