"""Module to test the grid interpolation."""

import numpy as np

import boxhi as bi
import unittest

class TestHI(bi.BoxHI):
    """
    This tests the C interpolation routine of the BoxHI class
    from sub_gridize_single_file down.
    It does this by generating a grid with known properties and then checking those properties are met
    load_header and set_nHI_grid are overloaded from the base class.

    Box is 2.5 Mpc.

    Note the neutral fraction calculation is NOT tested.
    """

    def __init__(self):
        print "Generating test grid..."
        self.generate_data()
        bi.BoxHI.__init__(self,"test",1,nslice=1,reload_file=True,savefile=None,gas=False, molec=False, ngrid=1638)

    def generate_data(self):
        """Generate a fake particle table. Need self.coords, self.mass and self.smooth"""
        cellsz = 2500/1638.
        #Length of a cell in physical cm
        cellszcm = (cellsz*3.08567758e21 / (1+3)/0.7)
        #Some particles in the middle of the grid, should be completely within a cell.
        #But only just completely within a cell
        self.smooth = np.ones(100, dtype=np.float32)*cellsz*0.49
        self.coords = np.zeros((100,3),dtype=np.float32)
        #Middle of the middle grid element: integrate over the x direction.
        self.coords[0:100,0] = np.arange(2,102)/1700.*2500.
        self.coords[0:100,1] = 839.5*2500/(1638.-1)
        self.coords[0:100,2] = 839.5*2500/(1638.-1)

        gadtonhi = cellszcm**2 * 1.67262178e-24 / (1e10 * 1.989e33/0.7)
        #Put this such that the total is slightly more than a DLA.
        self.mass = 10**(20.3) * gadtonhi/100.*np.ones(100,dtype=np.float32)/0.76


        #Stick a particle midway between four cells
        self.smooth = np.append(self.smooth, (cellsz*0.49))
        new = np.array([[10,739*2500/(1638.-1),839*2500/(1638.-1)]])
        self.coords = np.append(self.coords, new,0)
        self.mass = np.append(self.mass, 100*self.mass[-1])

        #Test a larger smoothing length
        self.smooth = np.append(self.smooth, np.repeat((cellsz),10))
        new = np.array([[10,700.5*2500/(1638.-1),700.5*2500/(1638.-1)]])
        self.coords = np.append(self.coords, np.repeat(new,10,0),0)
        self.mass = np.append(self.mass, np.repeat(10*self.mass[0],10))




    def load_header(self):
        """Load the header and halo data from a snapshot set"""
        #Simulation parameters
#         f=hdfsim.get_file(self.snapnum,self.snap_dir,0)
        self.redshift=3.
        self.hubble=0.7
        self.box=2500.
        self.npart=512 #f["Header"].attrs["NumPart_Total"]+2**32*f["Header"].attrs["NumPart_Total_HighWord"]
        self.omegam=0.27
        self.omegal=0.73
#         f.close()

    def set_nHI_grid(self, gas=False, location=0):
        """Set up the grid around each halo where the HI is calculated.
        """
#         star=cold_gas.RahmatiRT(self.redshift, self.hubble, molec=self.molec)
        self.once=True
        #Now grid the HI for each halo
        # files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
#         files.reverse()
#         for ff in files:
#             f = h5py.File(ff,"r")
#             print "Starting file ",ff
#             bar=f["PartType0"]
        ipos=self.coords #np.array(bar["Coordinates"])
        #Get HI mass in internal units
        mass=self.mass  #np.array(bar["Masses"])
        if not gas:
            #Hydrogen mass fraction
            mass *= self.hy_mass
            #mass *= star.get_reproc_HI(bar)
        smooth = self.smooth  #hsml.get_smooth_length(bar)
        [self.sub_gridize_single_file(ii,ipos,smooth,mass,self.sub_nHI_grid) for ii in xrange(0,self.nhalo)]
#         f.close()
        #Explicitly delete some things.
        #Deal with zeros: 0.1 will not even register for things at 1e17.
        #Also fix the units:
        #we calculated things in internal gadget /cell and we want atoms/cm^2
        #So the conversion is mass/(cm/cell)^2
        for ii in xrange(0,self.nhalo):
            massg=self.UnitMass_in_g/self.hubble/self.protonmass
            epsilon=2.*self.sub_radii[ii]/(self.ngrid[ii])*self.UnitLength_in_cm/self.hubble/(1+self.redshift)
            self.sub_nHI_grid[ii]*=(massg/epsilon**2)
            self.sub_nHI_grid[ii]+=0.1
            np.log10(self.sub_nHI_grid[ii],self.sub_nHI_grid[ii])
        return


class Tester(unittest.TestCase):
    def setUp(self):
        """Set grid correctly"""
        self.grid = testdata.sub_nHI_grid[0]

    def test_central_grid(self):
        """Check that the grid we get out is the one we expect."""
        #delta is 2.7 only...
        self.assertTrue(abs(self.grid[839,839]- 20.3) < 1e-6)
        self.assertAlmostEqual(np.sum(self.grid[838,837:840] + 1),0)
        self.assertAlmostEqual(np.sum(self.grid[840,837:840] + 1),0)
        self.assertAlmostEqual(np.sum(self.grid[837:840,837] + 1),0)
        self.assertAlmostEqual(np.sum(self.grid[837:840,840] + 1),0)

    def test_halfway_grid(self):
        """Particle between a few cells"""
        grid = testdata.sub_nHI_grid[0]
        self.assertTrue(np.all(self.grid[838:839,838:839]-20.3-np.log10(4) < 1e-6))

    def test_smooth_grid(self):
        """Particle between a few cells"""
        grid = testdata.sub_nHI_grid[0]
        #print self.grid[698:703,698:703]
        self.assertAlmostEqual(np.sum(self.grid[698,698:703] + 1),0)
        self.assertAlmostEqual(np.sum(self.grid[703,698:703] + 1),0)
        self.assertAlmostEqual(np.sum(self.grid[698:703,698] + 1),0)
        self.assertAlmostEqual(np.sum(self.grid[698:703,703] + 1),0)
        #Diagonals
        self.assertAlmostEqual(self.grid[699,699],self.grid[699,701])
        self.assertAlmostEqual(self.grid[701,699],self.grid[701,701])
        self.assertAlmostEqual(self.grid[701,699],15.0312144)
        #Cross-pieces
        self.assertAlmostEqual(self.grid[700,699],self.grid[700,701])
        self.assertAlmostEqual(self.grid[699,700],self.grid[699,700])
        self.assertAlmostEqual(self.grid[699,700],18.51635214)
        self.assertAlmostEqual(self.grid[700,700],20.27041661)

if __name__ == "__main__":
    #Make the test data global so it is only created once, not before every test.
    #Cheating, but whatever.
    testdata = TestHI()
    unittest.main()
