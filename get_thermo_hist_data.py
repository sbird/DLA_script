"""Script for getting the equation of state at z=3.2 from a variety of relevant simulations (on odyssey)"""
import gamma
import numpy as np
#outbase='/n/home11/spb/data/ComparisonProject/'
database='/home/spb/data/ComparisonProject/256_20Mpc/QUICK_LYA/'
# database='/n/hernquistfs1/mvogelsberger/ComparisonProject/'
#out=['Arepo_128/','Gadget_128/','Arepo_256/','Gadget_256/','Arepo/','Gadget/']
# base=['128_20Mpc/Arepo_ENERGY/output/','128_20Mpc/Gadget/output/','256_20Mpc/Arepo_ENERGY/output/','256_20Mpc/Gadget/output/','512_20Mpc/Arepo_ENERGY/output/','512_20Mpc/Gadget/output/']
base=['Arepo/output','Gadget/output']
# gam_list=[gamma.read_gamma(snap,database+base[0]) for snap in np.arange(0,12)]
# np.savetxt('/home/spb/scratch/ComparisonProject/Arepo_256_Qlya/thermo.txt', gam_list)
gam_list=[gamma.read_gamma(snap,database+base[1]) for snap in np.arange(0,10)]
np.savetxt('/home/spb/scratch/ComparisonProject/Gadget_256_Qlya/thermo.txt', gam_list)
