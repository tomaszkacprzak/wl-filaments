# WB00 - Wright Brainerd 2000

import pdb
import numpy as np
import os
import warnings
import cosmology
from scipy import integrate

cosmoparams = cosmology.cospars

class NfwHalo:

    def __init__(self):

        # self.scale_radius     =0      # characteristic radius of the cluster
        self.concentr           =5      # concentration parameter
        self.n00                =200.   # % N00 is 200 by default. If N00 = 500, this instead uses M500
        self.z_cluster          =0.3    # redshift to the cluster
        self.M_200              =9e14   # mass enclosed by 200 
        self.mn00type           ="crit" # don't remember
        self.theta_cx = 0               # center of the halo in arcmin
        self.theta_cy = 0               # center of the halo in arcmin
        self.redshift_offset = 0.2
        self.mean_inv_sigma_crit = None
        self.z_source = None
        
        self.update()


    # calculate the hidden variables
    def update(self):

        # get concentration
        self.concentr = self.get_concentr(method="Dutton")

        # cosmological critical density at redshift of the lens
        self.rho_crit_z_cluster = cosmoparams.rho_crit * (cosmoparams.Omega_m *(1+self.z_cluster)**3 + (1-cosmoparams.Omega_m))     
        # cosmological mean density at redshift of the lens
        self.rho_mean_z_cluster = cosmoparams.rho_crit * cosmoparams.Omega_m *(1+self.z_cluster)**3         

        self.delta_c = self.n00/3 * self.concentr**3 / ( np.log(1.+self.concentr) - self.concentr/(1.+self.concentr) ) # WB00eq2 

        if self.mn00type == "crit":

            self.rho = self.rho_crit_z_cluster 

        if self.mn00type == "mean":

            self.rho = self.rho_mean_z_cluster

        self.rho_s = self.delta_c * self.rho

        # calculate the radius of the n00
        # r_n00 = ( M_200 / (n00 * rho * 4/3 * pi ) ).^(1/3);
        self.r_n00 = ( np.abs(self.M_200) / (self.n00 * self.rho * 4./3. * np.pi ) )**(1./3.) 

        self.r_s = self.r_n00 / self.concentr 

        self.Dd  = cosmology.get_ang_diam_dist(self.z_cluster,0.)           # in units of h^-1 Mpc

        # calculate the radius in arcmin
        self.theta_n00 =  self.r_n00 / self.Dd *180./np.pi *60. 

        self.ang_diam_dist = cosmology.get_ang_diam_dist(self.z_cluster)


    def get_shear(self,mod_theta,z_source):

        self.update()

        # that may have been pre-calculated earlier
        if self.mean_inv_sigma_crit == None:
            
            warnings.warn('using re-calculated Sigma_crit')
            Ds  = cosmology.get_ang_diam_dist(z_source,0.)            # in units of h^-1 Mpc
            Dds = cosmology.get_ang_diam_dist(z_source,self.z_cluster)    # in units of h^-1 Mpc
            Dds_over_Ds = Dds / Ds

            # critical lens density
            Sigma_crit = cosmoparams.c**2 / (4* np.pi * cosmoparams.G) / Dds_over_Ds / self.Dd /cosmoparams.M_solar * cosmoparams.Mpc_to_m       
            # Sigma_crit = cosmology.get_sigma_crit(z_source,self.z_cluster)
            # print Sigma_crit
        else:
            warnings.warn('using pre-calculated Sigma_crit')
            Sigma_crit = 1./self.mean_inv_sigma_crit
        
        # dimensionless lens distance parameter
        x = mod_theta* 1./60. *np.pi/180. * self.Dd /self.r_s 

        # initialise arrays so they come out the right shape at the end
        kappa = np.zeros(x.shape);
        mod_gamma = np.zeros(x.shape);



        select = x < 1.
        kappa[select] =  2. * self.r_s * self.rho_s / Sigma_crit / (x[select]**2. -1.) * (1. - 2. / np.lib.scimath.sqrt(1. - x[select]**2) * np.arctanh(np.lib.scimath.sqrt( (1.-x[select]) / (1.+x[select]) )) )
        mod_gamma[select] = self.r_s * self.rho_s / Sigma_crit * ( 4./x[select]**2 * np.log(x[select]/2.) - 2./(x[select]**2-1) +  4. * np.arctanh(np.lib.scimath.sqrt((1.-x[select])/(1.+x[select]))) * (2. - 3.*x[select]**2) / ( x[select]**2 * np.lib.scimath.sqrt((1.-x[select]**2)**3)) )

        select = x == 1.
        kappa[select] =  2. * self.r_s * self.rho_s / Sigma_crit / 3.
        mod_gamma[select] = self.r_s * self.rho_s / Sigma_crit * ( 10./3. + 4.* np.log(0.5))

        select = x > 1.
        kappa[select] = 2. * self.r_s * self.rho_s / Sigma_crit / (x[select]**2 -1) * (1. - 2. / np.lib.scimath.sqrt(x[select]**2 -1) * np.arctan(np.lib.scimath.sqrt( (x[select]-1.) / (1.+x[select]) )) )
        

        # mod_mod_gamma(select) = real (r_s * rho_s / Sigma_crit * ( 4./x(select).^2 .* log(x(select)/2) - 2./(x(select).^2-1) +  4* atanh(sqrt((1-x(select))./(1+x(select)))) .* (2 - 3*x(select).^2)  ./ ( x(select).^2 .* (1-x(select).^2).^1.5 ) ) );
        
        mod_gamma[select] = (self.r_s * self.rho_s / Sigma_crit * ( 4./(x[select]**2) * np.log(x[select]/2.) - 2./(x[select]**2 - 1.) + 4.*np.arctanh(np.lib.scimath.sqrt( (1.-x[select])/(1.+x[select]))) * (2. - 3. * x[select]**2) / ( x[select]**2 * np.lib.scimath.power(1.-x[select]**2,1.5) ) ) )
        mod_gamma[select] = mod_gamma[select].real
     
        return (mod_gamma,kappa,Sigma_crit)

    def get_shears(self,theta_x,theta_y,z_source):


        # x position of each point on grid, relative to center
        dtheta_x=theta_x-self.theta_cx; 

        # y position of each point on grid, relative to center
        dtheta_y=theta_y-self.theta_cy; 

        # polar coords angle of each point on grid, from center, anticlockwise from +ve x axis
        theta=np.arctan2(dtheta_y,dtheta_x); 
        # clf; imagesc(theta_x,theta_y,angle); set(gca,'ydir','normal'); colorbar

        # polar coords distance of each point on grid, from center
        mod_theta=np.lib.scimath.sqrt( dtheta_x**2 + dtheta_y**2 );

        mod_gamma , kappa , Sigma_crit = self.get_shear(mod_theta,z_source);

        # convert to gamma_1 and gamma_2
        # gamma_1=-mod_gamma*np.cos(2*theta); # not sure how to justify minus sign..
        # gamma_2=-mod_gamma*np.sin(2*theta);
        # check
        #imagesc(theta_x,theta_y,gamma_1); set(gca,'ydir','normal'); colorbar
        #imagesc(theta_x,theta_y,gamma_2); set(gca,'ydir','normal'); colorbar

        # Fudge the difficult bits.
        # In practice won't have any galaxy shape estimates at the peak of the SIS
        # anyway.
        # gamma_1(find(isnan(gamma_1)))=0;
        # gamma_2(find(isnan(gamma_2)))=0;
        # kappa(find(isnan(kappa)))=0;

        # get the density contrast
        if self.M_200 >= 0:
            gamma_1=-mod_gamma*np.cos(2*theta); # not sure how to justify minus sign..
            gamma_2=-mod_gamma*np.sin(2*theta);
        else:
            gamma_2=-mod_gamma*np.cos(2*theta); # not sure how to justify minus sign..
            gamma_1=-mod_gamma*np.sin(2*theta);
            
        Delta_Sigma = kappa*Sigma_crit

        return gamma_1 , gamma_2 , Delta_Sigma, Sigma_crit , kappa

    def get_shears_with_pz(self,theta_x,theta_y , grid_z_centers , prob_z,  redshift_offset=0.2):

        list_h1g1 = []
        list_h1g2 = []
        list_weight = []

        for ib, vb in enumerate(grid_z_centers):
            if vb < (self.z_cluster+redshift_offset): continue
            [h1g1 , h1g2 , Delta_Sigma , Sigma_crit, kappa]= self.get_shears(theta_x,theta_y,vb)
            weight = prob_z[ib]
            list_h1g1.append(h1g1*weight) 
            list_h1g2.append(h1g2*weight) 
            list_weight.append(weight)

        h1g1 = np.sum(np.array(list_h1g1),axis=0) / np.sum(list_weight)
        h1g2 = np.sum(np.array(list_h1g2),axis=0) / np.sum(list_weight)

        return h1g1, h1g2

    def get_shears_with_pz_fast(self,theta_x,theta_y , grid_z_centers , prob_z,  redshift_offset=0.2):

        if self.z_source != None:

            [h1g1 , h1g2 , Delta_Sigma , Sigma_crit, kappa]= self.get_shears(theta_x,theta_y,self.z_source)

        else:

            [h1g1 , h1g2 , Delta_Sigma , Sigma_crit, kappa]= self.get_shears(theta_x,theta_y,None)

        return h1g1, h1g2 , Delta_Sigma, Sigma_crit, kappa

    def set_mean_inv_sigma_crit(self,grid_z_centers,prob_z,pair_z):

        sigma_crit = cosmology.get_sigma_crit(grid_z_centers,pair_z,unit='kg/m^2')
        prob_z_limit = prob_z
        prob_z_limit[grid_z_centers < pair_z + self.redshift_offset] = 0
        prob_z_limit /= sum(prob_z_limit)
        self.mean_inv_sigma_crit = sum(prob_z_limit / sigma_crit)

    def get_concentr(self,method="Duffy"):

        # Duffy et al 2008 from King and Mead 2011
        # concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cosmology.cospars.h)**(-0.081)
        if method=="Duffy":
            concentr = 5.72/(1.+self.z_cluster)**0.71 * (np.abs(self.M_200) / 1e14)**(-0.081)
        elif method=="Dutton":
            b = -0.101 + 0.026*self.z_cluster
            a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * self.z_cluster**1.21 )
            log10c = a + b*np.log10(np.abs(self.M_200)/10**12)
            concentr = 10**log10c
            # Dutton Maccio 2014 - Cold dark matter haloes in the Planck era -- evolution of structural parameters for Einasto and NFW profiles

        return concentr




if __name__=='__main__':
    
    nh = NfwHalo()
    nh.M_200=9e14
    nh.concentr=5
    nh.z_cluster=0.3
    z_source=0.8
    theta1 = np.linspace(-1,1,100)
    theta2 = np.linspace(-1,1,100)

    print nh.get_shears(theta1,theta2,z_source)

