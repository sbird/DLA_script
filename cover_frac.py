# vim: set fileencoding=utf-8
"""
Module for computing covering fractions, similar to how they are defined in Rudie et al. 2012 (e.g. see their Figure 15)
"""

import halohi
import numpy as np
import os.path as path
import math
import matplotlib.pyplot as plt
import matplotlib.colors

class CoverFrac(halohi.HaloHI):
    """
    Derived class with extra methods for plotting a pretty (high-resolution) picture of the grid around a halo.
    """

    def kpc_to_grid(self, kpc_dist, halo_num=0):
    	fac = self.ngrid[halo_num]/(2.*self.sub_radii[halo_num])
    	grid_dist = kpc_dist * fac
    	return grid_dist

    #def _dist(self,grid_cent,gridx,gridy)


	def halo_covering_frac(self, halo_num, N0_cutoffs, r_min_array, delta_r_kpc, return_cutoffs=False):
		halo_grid=np.array(self.sub_nHI_grid[halo_num])
		halo_radius = self.sub_radii[halo_num]

		#grid_len = np.size(halo_grid,axis=0)
		[gridx,gridy] = np.meshgrid(np.arange(ngrid),np.arange(ngrid))
		grid_cent = self.ngrid[halo_num]/2. #assume square grid: grid_centx = grid_centy = grid_cent

		delta_r_grid = self.kpc_to_grid(delta_r_kpc)
		r_grid = np.sqrt((gridx-grid_cent)**2+(gridy-grid_cent)**2)

		N_r_ticks = np.size(self.r_min_array)
		N_N0_ticks = np.size(N0_cutoffs)
		cover_frac = np.zeros([N_r_ticks,N_N0_ticks])
		
		for r_ind in np.arange(N_r_ticks):
			rmin = r_min_array[i]
			rmax = rmin + delta_r_grid

			rmin_cond = (r_grid >= rmin)
			rmax_cond = (r_grid < rmax)

			ind = np.where(np.logical_and(rmin_cond,rmax_cond))

			nHI_annulus = halo_grid[ind]
			n_tot = np.float(np.size(nHI_annulus))

			for N0_ind in np.arange(N_N0_ticks):
				if n_tot > 0:
					N0 = N0_cutoffs[j]
					n_covered = np.float(np.sum(nHI_annulus > N0))
					cover_frac[r_ind,N0_ind] = n_covered/n_tot
				elif n_tot == 0:
					cover_frac[r_ind,N0_ind] = -1.

		if return_cutoffs:
			return [cover_frac, r_min_array, N0_cutoffs]
		else:
			return [cover_frac, r_min_array]


	def aggregate_covering_fracs(self):
		topmass = 100.
		botmass = 10.

		mass_ind = np.where(np.logical_and(self.sub_mass > botmass, self.submass < topmass))
		n_halo = np.size(mass_ind)

		r_min_array = np.arange(0.,2900.,delta_r_kpc)
		N_r_ticks = np.size(self.r_min_array)

		N0_cutoffs = [13.0, 13.5, 14.0, 14.5, 15.0, 16.0]

		delta_r_kpc = 100.
		r_min_array = np.arange(0.,2900.,delta_r_kpc)

		Q1 = np.zeros(self.N_r_ticks)
		Q3 = np.zeros(self.N_r_ticks)
		for mass in massrange:
			#calculate cover frac for each halo 
			for N0_ind in np.arange(N_N0_ticks):
			#























