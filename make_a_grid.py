import sys
import halohi
import boxhi

partnum=str(sys.argv[3])
if sys.argv[1] == 'a':
    base="/home/spb/data/ComparisonProject/"+partnum+"_20Mpc/Arepo_ENERGY"
else:
    base="/home/spb/data/ComparisonProject/"+partnum+"_20Mpc/Gadget"

minpart=400
snapnum=sys.argv[2]


# 91,
# 124,
#]

base="/home/spb/data/finals/FINAL_E_NC/output"
# ahalo=halohi.HaloHI(base,snapnum,minpart,reload_file=True)
ahalo=boxhi.BoxHI(base,snapnum, reload_file=True)
ahalo.save_file()
