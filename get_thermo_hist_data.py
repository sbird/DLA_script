"""Script for getting the equation of state at z=3.2 from a variety of relevant simulations (on odyssey)"""
import gamma
#outbase='/n/home11/spb/data/ComparisonProject/'
database='/n/hernquistfs1/mvogelsberger/ComparisonProject/'
#out=['Arepo_128/','Gadget_128/','Arepo_256/','Gadget_256/','Arepo/','Gadget/']
base=['128_20Mpc/Arepo_ENERGY/output/','128_20Mpc/Gadget/output/','256_20Mpc/Arepo_ENERGY/output/','256_20Mpc/Gadget/output/','512_20Mpc/Arepo_ENERGY/output/','512_20Mpc/Gadget/output/']
gam_list=[gamma.read_gamma(124,database+bb) for bb in base]
