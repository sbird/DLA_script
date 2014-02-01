# -*- coding: utf-8 -*-
import sys
import boxhi
import halomet as hm
from os.path import expanduser

sim = "Cosmo"+str(sys.argv[2])+"_V6"
base=expanduser("~/data/Cosmo/"+sim+"/L25n512/output/")
# base=expanduser("~/data/Cosmo/"+sim+"/L25n256/")

snapnum=sys.argv[1]
outdir = base+"/snapdir_"+str(snapnum).rjust(3,'0')

# 91,
# 124,
#]
ahalo=hm.BoxCIV(base,snapnum, reload_file=True, nslice=30,savefile=outdir+"/boxciv_grid_low.hdf5",ngrid=5000)
ahalo.column_density_function()
ahalo.rho_DLA(13.)
ahalo.line_density(13.)
ahalo.omega_DLA(13.,15,1e8)
print ahalo.omega_DLA(13.,15,1e8)
ahalo.save_file(LLS_cut=11.5, DLA_cut = 13.)
