import sys
import halohi

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

# ahalo=halohi.HaloHI(base,snapnum,minpart)
ahalo=halohi.VelocityHI(base,snapnum,minpart)
ahalo.save_file()
# ahalo=halohi.TotalHaloHI(base,snapnum,minpart)
# ahalo.save_file()
