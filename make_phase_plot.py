"""Script for making Lyman-alpha forest temperature-density plots using the phase_plot module"""

import matplotlib
matplotlib.use('PDF')

import phase_plot
import matplotlib.pyplot as plt
import numpy as np
outdir="/home/spb/scratch/ComparisonProject/"
bar_label="Mass ($10^{6} M_\odot$ h$^{-1}$)"
### 256**3 particles
gad=phase_plot.get_mass_map(124,"/home/spb/data/ComparisonProject/256_20Mpc/Gadget")
ar=phase_plot.get_mass_map(124,"/home/spb/data/ComparisonProject/256_20Mpc/Arepo_ENERGY")

#Make some plots
plt.imshow(gad.map,origin='lower',extent=gad.get_lims(),aspect='auto',vmin=0,vmax=30)
bar=plt.colorbar(use_gridspec=True)
bar.set_label(bar_label)
plt.xticks((3.5,4.0,4.5,5.0))
plt.xlabel(r"$\log_{10}\, T \,(10^4 K)$")
plt.ylabel(r"$\log_{10} \left(\rho /\rho_\mathrm{c}\right)$")
plt.tight_layout()
plt.savefig(outdir+"phase_gad_256.pdf")
plt.figure()
plt.imshow(ar.map,origin='lower',extent=ar.get_lims(),aspect='auto',vmin=0,vmax=30)
bar=plt.colorbar(use_gridspec=True)
bar.set_label(bar_label)
plt.xticks((3.5,4.0,4.5,5.0))
plt.xlabel(r"$\log_{10}\, T \,(10^4 K)$")
plt.ylabel(r"$\log_{10} \left(\rho /\rho_\mathrm{c}\right)$")
plt.tight_layout()
plt.savefig(outdir+"phase_ar_256.pdf")
plt.figure()
plt.imshow(gad.map-ar.map,origin='lower',extent=gad.get_lims(),aspect='auto',vmin=-30,vmax=30)
bar=plt.colorbar(use_gridspec=True)
bar.set_label(bar_label)
plt.xticks((3.5,4.0,4.5,5.0))
plt.xlabel(r"$\log_{10}\, T \,(10^4 K)$")
plt.ylabel(r"$\log_{10} \left(\rho /\rho_\mathrm{c}\right)$")
plt.tight_layout()
plt.savefig(outdir+"phase_gad_ar_256.pdf")
plt.figure()

### 512**3 particles
ar_512=phase_plot.get_mass_map(124,"/home/spb/data/ComparisonProject/512_20Mpc/Arepo_ENERGY")
gad_512=phase_plot.get_mass_map(124,"/home/spb/data/ComparisonProject/512_20Mpc/Gadget")

plt.imshow(gad_512.map,origin='lower',extent=gad_512.get_lims(),aspect='auto',vmin=0,vmax=30)
bar=plt.colorbar(use_gridspec=True)
bar.set_label(bar_label)
plt.xticks((3.5,4.0,4.5,5.0))
plt.xlabel(r"$\log_{10}\, T \,(10^4 K)$")
plt.ylabel(r"$\log_{10} \left(\rho /\rho_\mathrm{c}\right)$")
plt.tight_layout()
plt.savefig(outdir+"phase_gad_512.pdf")
plt.figure()
plt.imshow(ar_512.map,origin='lower',extent=ar_512.get_lims(),aspect='auto',vmin=0,vmax=30)
bar=plt.colorbar(use_gridspec=True)
bar.set_label(bar_label)
plt.xticks((3.5,4.0,4.5,5.0))
plt.xlabel(r"$\log_{10}\, T \,(10^4 K)$")
plt.ylabel(r"$\log_{10} \left(\rho /\rho_\mathrm{c}\right)$")
plt.tight_layout()
plt.savefig(outdir+"phase_ar_512.pdf")
plt.figure()
plt.imshow(gad_512.map-ar_512.map,origin='lower',extent=gad_512.get_lims(),aspect='auto',vmin=-30,vmax=30)
bar=plt.colorbar(use_gridspec=True)
bar.set_label(bar_label)
plt.xticks((3.5,4.0,4.5,5.0))
plt.xlabel(r"$\log_{10}\, T \,(10^4 K)$")
plt.ylabel(r"$\log_{10} \left(\rho /\rho_\mathrm{c}\right)$")
plt.tight_layout()
plt.savefig(outdir+"phase_gad_ar_512.pdf")

#arepo="/home/spb/data/ComparisonProject/256_20Mpc/Arepo_ENERGY"
#gadget="/home/spb/data/ComparisonProject/256_20Mpc/Gadget"
#(atlg, aoden,amass)=phase_plot.get_temp_overden_mass(124,arepo)
#gtlg=np.array([])
#goden=np.array([])
#gmass=np.array([])
#gind=np.array([])
##Select particles where there are not many particles in Gadget
#ind=np.where((aoden> -0.01)*(aoden < 0.01)*(atlg > 4.49)*(atlg < 4.51))
#ind2=np.where((aoden> -0.53)*(aoden < -0.51)*(atlg > 3.89)*(atlg < 3.91))
#
#for p in np.append(np.ravel(ind),np.ravel(ind2)):
#    aid=phase_plot.find_id(p,124,arepo)
#    try:
#        (part,snap_file)=phase_plot.find_particle(aid,124,gadget)
#    except IOError:
#        continue
#    (gtlog, godensit,gmss)=phase_plot.get_temp_overden_mass(124,gadget,snap_file)
#    gtlg=np.append(gtlg,gtlog[part])
#    goden=np.append(goden,godensit[part])
#    gmass=np.append(gmass,gmss[part])
#    gind=np.append(gind,p)
#
#

