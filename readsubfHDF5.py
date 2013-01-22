# Python HDF5 subfind reader 
# (requires PyTables http://www.pytables.org)
#
# import readsubfHDF5
# cat = readsubfHDF5.subfind_catalog("./output/", 60)
#
# Mark Vogelsberger (mvogelsb@cfa.harvard.edu)

import numpy as np
import os
import sys
import tables

####################
#SUBHALO DATABLOCKS#
####################
#descriptions of subhalo datablocks -> add new datablocks here!
#format -> "HDF5_NAME":["DATATYPE", DIMENSION]
sub_datablocks = {"SubhaloLen":["INT",1],
                  "SubhaloMass":["FLOAT",1],
                  "SubhaloMassinRad":["FLOAT",1],
                  "SubhaloPos":["FLOAT",3],
                  "SubhaloVel":["FLOAT",3],
                  "SubhaloLenType":["INT",6],
                  "SubhaloMassType":["FLOAT",6],
                  "SubhaloCM":["FLOAT",3],
                  "SubhaloSpin":["FLOAT",3],
                  "SubhaloVelDisp":["FLOAT",1],
                  "SubhaloVmax":["FLOAT",1],
                  "SubhaloVmaxRad":["FLOAT",1],
                  "SubhaloHalfmassRad":["FLOAT",1],
                  "SubhaloHalfmassRadType":["FLOAT",6],
                  "SubhaloMassInRadType":["FLOAT", 6],
                  "SubhaloIDMostbound":["ID",1],
                  "SubhaloGrNr":["INT",1],
                  "SubhaloParent":["INT",1],
		  "SubhaloSFR":["FLOAT",1],
                  "SubhaloSFRinRad":["FLOAT",1],
                  "SubhaloGasMetallicity":["FLOAT",1],
                  "SubhaloGasMetallicitySfr":["FLOAT",1],
                  "SubhaloStarMetallicity":["FLOAT",1],
                  "SubhaloGasMetalFractions":["FLOAT",9],
                  "SubhaloGasMetalFractionsSfr":["FLOAT",9],
                  "SubhaloGasMetalFractionsSfrWeighted":["FLOAT",9],
                  "SubhaloStarMetalFractions":["FLOAT",9],
                  "SubhaloBHMass":["FLOAT",1],
                  "SubhaloBHMdot":["FLOAT",1],
                  "SubhaloStellarPhotometrics":["FLOAT",8]}  #band luminosities: U, B, V, K, g, r, i, z

##################
#GROUP DATABLOCKS#
##################
#descriptions of subhalo datablocks -> add new datablocks here!
#format -> "HDF5_NAME":["DATATYPE", DIMENSION]
grp_datablocks = {"GroupLen":["INT",1],
                  "GroupMass":["FLOAT",1],
                  "GroupPos":["FLOAT",3],
                  "GroupVel":["FLOAT",3],
                  "GroupLenType":["INT",6],
                  "GroupMassType":["FLOAT",6],
                  "Group_M_Mean200":["FLOAT",1],
                  "Group_R_Mean200":["FLOAT",1],
                  "Group_M_Crit200":["FLOAT",1],
                  "Group_R_Crit200":["FLOAT",1],
                  "Group_M_TopHat200":["FLOAT",1],
                  "Group_R_TopHat200":["FLOAT",1],
                  "GroupNsubs":["INT",1],
                  "GroupFirstSub":["INT",1],
                  "GroupSFR":["FLOAT",1],
                  "GroupGasMetallicity":["FLOAT",1],
                  "GroupStarMetallicity":["FLOAT",1],
                  "GroupGasMetalFractions":["FLOAT",9],
                  "GroupStarMetalFractions":["FLOAT",9],
                  "GroupBHMass":["FLOAT",1],
                  "GroupBHMdot":["FLOAT",1], 
		  "GroupFuzzOffsetType":["INT64",6]}

class subfind_catalog:
	def __init__(self, basedir, snapnum, long_ids = False, double_output = False, grpcat = True, subcat = True, name = "fof_subhalo_tab"):
		self.filebase = basedir + "/groups_" + str(snapnum).zfill(3) + "/" + name + "_" + str(snapnum).zfill(3) + "."
 
		if long_ids: self.id_type = np.uint64
		else: self.id_type = np.uint32
		if double_output: self.double_type = np.float32
		else: self.double_type = np.float64

		filenum = 0
		doneflag = False
		skip_gr = 0
		skip_sub = 0
		vardict = {}

		while not doneflag:
			curfile = self.filebase + str(filenum) + ".hdf5"

			if (not os.path.exists(curfile)):
				self.filebase = basedir + "/" + name + "_" + str(snapnum).zfill(3)
				curfile = self.filebase + ".hdf5"
			if (not os.path.exists(curfile)):
				print "file not found:", curfile
				sys.exit()

			f=tables.openFile(curfile)      
			ngroups = f.root.Header._v_attrs.Ngroups_ThisFile
			nsubs = f.root.Header._v_attrs.Nsubgroups_ThisFile
			nfiles = f.root.Header._v_attrs.NumFiles
			if filenum == 0:
				self.ngroups = f.root.Header._v_attrs.Ngroups_Total 
				self.nids = f.root.Header._v_attrs.Nids_Total 
				self.nsubs = f.root.Header._v_attrs.Nsubgroups_Total 
				#GROUPS
				if (grpcat==True):
				       	for key, val in grp_datablocks.items():
						if f.root.Group.__contains__(key): 
					                type = val[0]
        					        dim = val[1]
				                	if (type=='FLOAT'):
		        			                vars(self)[key]=np.empty(self.ngroups, dtype=np.dtype((self.double_type,dim)))
			        		        if (type=='INT'):
	                				        vars(self)[key]=np.empty(self.ngroups, dtype=np.dtype((np.int32,dim)))
				                        if (type=='INT64'):
        	        			                vars(self)[key]=np.empty(self.ngroups, dtype=np.dtype((np.int64,dim)))
			        		        if (type=='ID'):
                					        vars(self)[key]=np.empty(self.ngroups, dtype=np.dtype((self.id_type,dim)))
							vardict[key]=vars(self)[key]
				#SUBHALOS
				if (subcat==True):
					for key, val in sub_datablocks.items():
						if f.root.Subhalo.__contains__(key):
				                        type = val[0]
			        	                dim = val[1]
						if (type=='FLOAT'):
							vars(self)[key]=np.empty(self.nsubs, dtype=np.dtype((self.double_type,dim)))
						if (type=='INT'):
		        		        	vars(self)[key]=np.empty(self.nsubs, dtype=np.dtype((np.int32,dim)))
		                	        if (type=='INT64'):
		                        	        vars(self)[key]=np.empty(self.nsubs, dtype=np.dtype((np.int32,dim)))
				                if (type=='ID'):
        				                vars(self)[key]=np.empty(self.nsubs, dtype=np.dtype((self.id_type,dim)))
						vardict[key]=vars(self)[key]

		      	#GROUPS
      			if (grpcat==True):
				if ngroups > 0:
		                	for key, val in grp_datablocks.items():
						if f.root.Group.__contains__(key):
			                        	type = val[0]
	                			        dim = val[1]
			                                a=f.root.Group._f_getChild(key)
                        			        if dim==1:
			                                        vardict[key][skip_gr:skip_gr + ngroups]=a[:]
			                                else:
                        			                for d in range(0,dim):
			                                                vardict[key][skip_gr:skip_gr + ngroups,d]=a[:,d]
					skip_gr += ngroups
			#SUBHALOS 
			if (subcat==True): 
				if nsubs > 0:
					for key, val in sub_datablocks.items():
						if f.root.Subhalo.__contains__(key):
							type = val[0]
							dim = val[1]
							a=f.root.Subhalo._f_getChild(key)
							if dim==1:
								vardict[key][skip_sub:skip_sub + nsubs]=a[:]
							else:
								for d in range(0,dim):
									vardict[key][skip_sub:skip_sub + nsubs,d]=a[:,d]
					skip_sub += nsubs      

			f.close()

			filenum += 1
			if filenum == nfiles: doneflag = True
