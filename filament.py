import pdb
import numpy as np
import pylab as pl
import os
import warnings
import cosmology
from scipy import integrate
import plotstools, tabletools
from scipy.interpolate import interp1d

cosmoparams = cosmology.cospars

class filament:

    def __init__(self):

        self.grid_z_edges = None
        self.grid_z_centers = None
        self.prob_z = None
        self.pair_z = None
        self.redshift_offset = 0.2
        self.mean_inv_sigma_crit = None

    def proj_mass_density(self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc):

        ## new ## ---------------------
        # amplitude = kappa0 /  (radius_mpc * np.pi) 

        # r = np.abs(shear_v_mpc)

        # dens = amplitude / (1. + (r / radius_mpc)**2 )

        # # zero the filament outside halos
        # # we shoud zero it at R200, but I have to check how to calculate it from M200 and concentr
        # select = ((shear_u_mpc > (u1_mpc) ) + (shear_u_mpc < (u2_mpc)  ))
        # dens[select] *= 0.

        # amplitude = kappa0 /  (radius_mpc * np.pi) 
        r = np.abs(shear_v_mpc)
        dens = kappa0 / (1. + (r / radius_mpc)**2 )

        # # zero the filament outside halos
        # # we shoud zero it at R200, but I have to check how to calculate it from M200 and concentr
        select = ((shear_u_mpc > (u1_mpc) ) + (shear_u_mpc < (u2_mpc)  ))
        dens[select] *= 0.


        return dens

    def get_filament_model_interp(self,kappa0,radius_mpc,pair_z, grid_z_centers,prob_z):

        radius_min = 0
        radius_max = 10
        v = np.linspace(radius_min,radius_max,1000)
        u = np.zeros_like(v)
       
        # (self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z, grid_z_centers , prob_z,  redshift_offset=0.2):
        g1 , g2 = self.filament_model_with_pz( u, v, -1, 1, kappa0, radius_mpc, pair_z, grid_z_centers, prob_z)
        pmdi = interp1d( v , g1 , 'nearest')

        return pmdi


    def filament_model(self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z,shear_z):
        """
        @brief create filament model
        @param shear_u_mpc array of u coordinates
        @param shear_v_mpc array of v coordinates
        @param u1_mpc start of the filament 
        @param u2_mpc end of the filament 
        @param kappa0 total mass of the filament 
        """

        dens = self.proj_mass_density(shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc)
        sigma_crit = cosmology.get_sigma_crit(shear_z,pair_z,unit='Msol*h/pc^2')
        # print 'sigma_crit' , sigma_crit
        # print 'max dens' , max(dens)
        kappa = dens / sigma_crit 
        g1 = -kappa
        g2 = np.zeros_like(kappa)

        return g1 , g2


    def set_mean_inv_sigma_crit(self,grid_z_centers,prob_z,pair_z):

        sigma_crit = cosmology.get_sigma_crit(grid_z_centers,pair_z,unit='Msol*h/pc^2')
        prob_z_limit = prob_z
        prob_z_limit[grid_z_centers < pair_z + self.redshift_offset] = 0
        prob_z_limit /= sum(prob_z_limit)
        self.mean_inv_sigma_crit = sum(prob_z_limit / sigma_crit)


    def filament_model_with_pz(self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z, grid_z_centers , prob_z,  redshift_offset=0.2):

        dens = self.proj_mass_density(shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc)
        kappa = dens * self.mean_inv_sigma_crit 
        g1 = -kappa
        g2 = np.zeros_like(kappa)

        return g1 , g2


    def get_bcc_pz(self):

        self.grid_z_edges = np.linspace(0,2,10);
        self.grid_z_centers = plotstools.get_bins_centers(self.grid_z_edges)

        if self.prob_z == None:

            filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
            lenscat = tabletools.loadTable(filename_lenscat)
            self.prob_z , _  = pl.histogram(lenscat['z'],bins=self.grid_z_edges,normed=True)




if __name__=='__main__':

    # mass = np.logspace(0,15,100)
    mass = np.linspace(0, 4e2 ,10)
    n_points = 100
    shear_u_mpc = np.ones(n_points)*0
    shear_v_mpc = np.linspace(-10,10,n_points)
    u1_mpc = 10
    u2_mpc = -10
    kappa0 = None
    radius_mpc = 1

    f = filament()
    f.pair_z = 0.6
    f.n_points = len(shear_u_mpc)
    f.redshift_offset = 0.2

    f.get_bcc_pz()
    f.set_mean_inv_sigma_crit(f.grid_z_centers,f.prob_z,f.pair_z)


    kappa0 = 50
    radius_mpc = 0.5
    model_pz_1 , model_pz_2 = f.filament_model_with_pz( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'ro')
    print min(model_pz_1)

    kappa0 = 150
    radius_mpc = 3
    model_pz_1 , model_pz_2 = f.filament_model_with_pz( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'bx',ms=20)
    print min(model_pz_1)

    kappa0 = 150
    radius_mpc = 0.5
    model_pz_1 , model_pz_2 = f.filament_model_with_pz( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'gd')
    print min(model_pz_1)

    kappa0 = 50
    radius_mpc = 3
    model_pz_1 , model_pz_2 = f.filament_model_with_pz( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'m+',ms=20)
    print min(model_pz_1)

    pl.yscale('log')
    pl.show()






