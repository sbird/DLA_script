# -*- coding: utf-8 -*-
import sys
import boxhi
import halomet as hm
from os.path import expanduser

sim = "Cosmo"+str(sys.argv[2])+"_V6"
base=expanduser("~/data/Cosmo/"+sim+"/L25n512/output/")

snapnum=sys.argv[1]
outdir = base+"/snapdir_"+str(snapnum).rjust(3,'0')

# 91,
# 124,
#]
ahalo=hm.BoxCIV(base,snapnum, reload_file=True, nslice=30,savefile=outdir+"/boxciv_grid_big.hdf5",ngrid=10000)
ahalo.column_density_function()
ahalo.rho_DLA(14.)
ahalo.line_density(14.)
ahalo.omega_DLA(14.)
print ahalo.omega_DLA(14.)
ahalo.save_file(LLS_cut=13., DLA_cut = 14.)
