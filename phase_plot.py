import hdfsim
import math
import numpy as np

class mass_map:
        def __init__(self):
                self.minT=3.2
                self.maxT=6.0
                self.minR=-1.5
                self.maxR=1.0
                self.map = np.zeros([300,300]) 
                self.r_size=np.shape(self.map)[0]
                self.t_size=np.shape(self.map)[1]

        def rho_ind(self,rho):
                x=(rho-self.minR)/(self.maxR-self.minR)*(self.r_size-1)
                return np.array(np.around(x),dtype=int)

        def temp_ind(self,temp):
                x=(temp-self.minT)/(self.maxT-self.minT)*(self.t_size-1)
                return np.array(np.around(x),dtype=int)

        def add_to_map(self,rho, temp,mass):
                self.map[self.rho_ind(rho),self.temp_ind(temp)]+=mass

        def get_rho(self):
                return (self.maxR-self.minR)*np.arange(0,self.r_size)/(self.r_size-1)+self.minR

        def get_temp(self):
                return (self.maxT-self.minT)*np.arange(0,self.t_size)/(self.t_size-1)+self.minT

        def get_lims(self):
                return(self.minT,self.maxT,self.minR,self.maxR)

def get_temp_overden_mass(num,base,file=0):
        f=hdfsim.get_file(num,base,file)
#         print 'Reading file from:',fname
        
        head=f["Header"].attrs
        npart=head["NumPart_ThisFile"]
        redshift=head["Redshift"]
        atime=head["Time"]
        h100=head["HubbleParam"]

        if npart[0] == 0 :
                print "No gas particles!\n"
                return
        
        # Baryon density parameter
        omegab0 = 0.0449
        # Scaling factors and constants
        Xh = 0.76                       # Hydrogen fraction
        G = 6.672e-11                   # N m^2 kg^-2
        kB = 1.3806e-23                 # J K^-1
        Mpc = 3.0856e22                 # m
        kpc = 3.0856e19                 # m
        Msun = 1.989e30                 # kg
        mH = 1.672e-27                  # kg
        H0 = 1.e5/Mpc                   # 100 km s^-1 Mpc^-1 in SI units
        gamma = 5.0/3.0

        rscale = (kpc * atime)/h100     # convert length to m
        vscale = atime**0.5              # convert velocity to km s^-1
        mscale = (1e10 * Msun)/h100     # convert mass to kg
        dscale = mscale / (rscale**3.0)  # convert density to kg m^-3 
        escale = 1e6                    # convert energy/unit mass to J kg^-1
        
        bar = f["PartType0"]
        u=escale*np.array(bar['InternalEnergy'],dtype=np.float64) # J kg^-1
        rho=dscale*np.array(bar['Density'],dtype=np.float64) # kg m^-3, ,physical
        mass=np.array(bar['Masses'],dtype=np.float64)
        nelec=np.array(bar['ElectronAbundance'],dtype=np.float64)
        nH0=np.array(bar['NeutralHydrogenAbundance'],dtype=np.float64)
        f.close()
        # Convert to physical SI units. Only energy and density considered here.                 
        ## Mean molecular weight
        mu = 1.0 / ((Xh * (0.75 + nelec)) + 0.25)  
        templog=np.log10(mu/kB * (gamma-1) * u * mH)
        ##### Critical matter/energy density at z=0.0
        rhoc = 3 * (H0*h100)**2 / (8. * math.pi * G) # kg m^-3
        ##### Mean hydrogen density of the Universe
        nHc = rhoc  /mH * omegab0 *Xh * (1.+redshift)**3.0 
        ### Hydrogen density as a fraction of the mean hydrogen density
        overden = np.log10(rho*Xh/mH  / nHc)

        return (templog,overden,mass)

def get_temp_overden_volume(num,base,file=0):
        f=hdfsim.get_file(num,base,file)
#         print 'Reading file from:',fname
        
        head=f["Header"].attrs
        npart=head["NumPart_ThisFile"]
        redshift=head["Redshift"]
        atime=head["Time"]
        h100=head["HubbleParam"]

        if npart[0] == 0 :
                print "No gas particles!\n"
                return
        
        # Baryon density parameter
        omegab0 = 0.0449
        # Scaling factors and constants
        Xh = 0.76                       # Hydrogen fraction
        G = 6.672e-11                   # N m^2 kg^-2
        kB = 1.3806e-23                 # J K^-1
        Mpc = 3.0856e22                 # m
        kpc = 3.0856e19                 # m
        Msun = 1.989e30                 # kg
        mH = 1.672e-27                  # kg
        H0 = 1.e5/Mpc                   # 100 km s^-1 Mpc^-1 in SI units
        gamma = 5.0/3.0

        rscale = (kpc * atime)/h100     # convert length to m
        vscale = atime**0.5              # convert velocity to km s^-1
        mscale = (1e10 * Msun)/h100     # convert mass to kg
        dscale = mscale / (rscale**3.0)  # convert density to kg m^-3 
        escale = 1e6                    # convert energy/unit mass to J kg^-1
        
        bar = f["PartType0"]
        u=escale*np.array(bar['InternalEnergy'],dtype=np.float64) # J kg^-1
        rho=dscale*np.array(bar['Density'],dtype=np.float64) # kg m^-3, ,physical
        mass=np.array(bar['Masses'],dtype=np.float64)
        nelec=np.array(bar['ElectronAbundance'],dtype=np.float64)
        #nH0=np.array(bar['NeutralHydrogenAbundance'],dtype=np.float64)
         
        f.close()
        # Convert to physical SI units. Only energy and density considered here.                 
        ## Mean molecular weight
        mu = 1.0 / ((Xh * (0.75 + nelec)) + 0.25)  
        templog=np.log10(mu/kB * (gamma-1) * u * mH)
        ##### Critical matter/energy density at z=0.0
        rhoc = 3 * (H0*h100)**2 / (8. * math.pi * G) # kg m^-3
        ##### Mean hydrogen density of the Universe
        nHc = rhoc  /mH * omegab0 *Xh * (1.+redshift)**3.0 
        ### Hydrogen density as a fraction of the mean hydrogen density
        overden = np.log10(rho*Xh/mH  / nHc)
        volume=dscale*mass/rho
        rad=(3*volume/(4*math.pi))**(1/3.)
        return (rad,overden,mass)


def get_mass_map(num,base):
        masses=mass_map()
        for i in np.arange(0,500) :
                try:
                        (templog,overden,mass) = get_temp_overden_mass(num,base,i)
                except IOError:
                        break
                ind2 = np.where((overden > masses.minR) * (overden <  masses.maxR) * (templog < masses.maxT) * (templog > masses.minT))
                masses.add_to_map(overden[ind2],templog[ind2],mass[ind2])
        return masses

def get_volume_forest(num,base):
        masses=mass_map()
        masses.minT=10
        masses.maxT=150
        for i in np.arange(0,500) :
                try:
                        (rad,overden,mass) = get_temp_overden_volume(num,base,i)
                except IOError:
                        break
                ind2 = np.where((overden > -1.5) * (overden <  0.0))
                masses.add_to_map(overden[ind2],rad[ind2],mass[ind2])
                return masses
                #Return radius
                #return (overden[ind2],templog[ind2],(3*volume[ind2]/(4*math.pi))**(1/3.))

def find_particle(part_id,num,base):
        """Gets the file and position of a particle with id part_id"""
        for i in range(0,500) :
                ids=hdfsim.get_baryon_array("ParticleIDs",num,base,i,dtype=np.int64)
                ind=np.where(ids==part_id)
                if np.size(ind) > 0:
                        return (ind[0][0], i)

def find_id(pos,num,base,file=0):
        """Gets the id of a particle at some position"""
        ids=hdfsim.get_baryon_array("ParticleIDs",num,base,file,dtype=np.int64)
        return ids[pos]

"""
### 256**3 particles
gad=phase_plot.get_mass_map(124,"/home/spb/data/ComparisonProject/256_20Mpc/Gadget")
ar=phase_plot.get_mass_map(124,"/home/spb/data/ComparisonProject/256_20Mpc/Arepo_ENERGY")

#Make some plots
plt.imshow(gad.map,origin='lower',extent=gad.get_lims(),aspect='auto',vmin=0,vmax=0.003)
plt.colorbar()
figure()
plt.imshow(ar.map,origin='lower',extent=ar.get_lims(),aspect='auto',vmin=0,vmax=0.003)
plt.colorbar()
figure()
imshow(gad.map-ar.map,origin='lower',extent=gad.get_lims(),aspect='auto',vmin=-0.003,vmax=0.003)
plt.colorbar()
figure()

### 512**3 particles
ar_512=phase_plot.get_mass_map(124,"/home/spb/data/ComparisonProject/512_20Mpc/Arepo_ENERGY")
gad_512=phase_plot.get_mass_map(124,"/home/spb/data/ComparisonProject/512_20Mpc/Gadget")

plt.imshow(gad_512.map,origin='lower',extent=gad_512.get_lims(),aspect='auto',vmin=0,vmax=0.003)
plt.colorbar()
figure()
plt.imshow(ar_512.map,origin='lower',extent=ar_512.get_lims(),aspect='auto',vmin=0,vmax=0.003)
plt.colorbar()
figure()
imshow(gad_512.map-ar_512.map,origin='lower',extent=gad_512.get_lims(),aspect='auto',vmin=-0.003,vmax=0.003)
plt.colorbar()


"""

"""
arepo="/home/spb/data/ComparisonProject/256_20Mpc/Arepo_ENERGY"
gadget="/home/spb/data/ComparisonProject/256_20Mpc/Gadget"
(atlg, aoden,amass)=phase_plot.get_temp_overden_mass(124,arepo)
gtlg=np.array([])
goden=np.array([])
gmass=np.array([])
gind=np.array([])
#Select particles where there are not many particles in Gadget
ind=np.where((aoden> -0.01)*(aoden < 0.01)*(atlg > 4.49)*(atlg < 4.51))
ind2=np.where((aoden> -0.53)*(aoden < -0.51)*(atlg > 3.89)*(atlg < 3.91))

for p in np.append(np.ravel(ind),np.ravel(ind2)):
        aid=phase_plot.find_id(p,124,arepo)
        try:
                (part,file)=phase_plot.find_particle(aid,124,gadget)
        except IOError:
                continue
        (gtlog, godensit,gmss)=phase_plot.get_temp_overden_mass(124,gadget,file)
        gtlg=np.append(gtlg,gtlog[part])
        goden=np.append(goden,godensit[part])
        gmass=np.append(gmass,gmss[part])
        gind=np.append(gind,p)

"""

 
