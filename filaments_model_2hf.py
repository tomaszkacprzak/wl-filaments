import os, yaml, argparse, sys, logging , pyfits,  emcee, tabletools, cosmology, filaments_tools, nfw, plotstools, filament, time
import numpy as np
import pylab as pl
import warnings
import scipy.interpolate

warnings.simplefilter('once')

log = logging.getLogger("fil_mod_2hf") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

redshift_offset = 0.2
weak_limit = 0.1

dtype_stats = {'names' : ['id',
                            'kappa0_signif',
                            'kappa0_map',
                            'kappa0_err_hi',
                            'kappa0_err_lo',
                            'radius_map',
                            'radius_err_hi',
                            'radius_err_lo',
                            'h1M200_map',
                            'h1M200_err_hi',
                            'h1M200_err_lo',
                            'h2M200_map',
                            'h2M200_err_hi',
                            'h2M200_err_lo',
                            'chi2_red_null',
                            'chi2_red_max',
                            'chi2_red_D',
                            'chi2_red_LRT' ,
                            'chi2_null',
                            'chi2_max',
                            'chi2_D' ,
                            'chi2_LRT' ,
                            'sigma_g' ] ,
                'formats' : ['i8'] + ['f8']*22 }


class modelfit():

    def __init__(self):
        

        self.grid_z_edges = np.linspace(0,2,10);
        self.grid_z_centers = plotstools.get_bins_centers(self.grid_z_edges)
        self.prob_z = None
        self.sigma_g = 0.2
        self.inv_sq_sigma_g = None
        self.n_model_evals = 0
        self.sampler = None
        self.n_samples = 1000
        self.n_walkers = 10
        self.n_grid = 10
        self.save_all_models = False
        self.kappa_is_K = False

        self.n_dim = 4 
        self.parameters = [None]*self.n_dim
        self.parameters[0] = {}
        self.parameters[1] = {}
        self.parameters[2] = {}
        self.parameters[3] = {}
        
        self.parameters[0]['name'] = 'filament_kappa' # density contrast
        self.parameters[1]['name'] = 'filament_radius' # Mpc
        self.parameters[2]['name'] = 'halo1_M200' # log10(M_solar)
        self.parameters[3]['name'] = 'halo2_M200' # log10(M_solar)

        self.parameters[0]['box'] = {'min' : 0,  'max': 150}
        self.parameters[1]['box'] = {'min' : 0,  'max': 10}
        self.parameters[2]['box'] = {'min' : 13, 'max': 15}
        self.parameters[3]['box'] = {'min' : 13, 'max': 15}

        self.shear_g1 = None
        self.shear_g2 = None
        self.shear_n_gals = None
        self.shear_u_arcmin = None
        self.shear_v_arcmin = None
        self.shear_u_mpc = None
        self.shear_v_mpc = None
        self.halo1_u_arcmin = None
        self.halo1_v_arcmin = None
        self.halo1_u_mpc = None
        self.halo1_v_mpc = None
        self.halo1_z = None
        self.halo2_u_arcmin = None
        self.halo2_v_arcmin = None
        self.halo2_u_mpc = None
        self.halo2_v_mpc = None
        self.halo2_z = None
        
        # self.halo1_M200 = None
        # self.halo1_conc = None      
        # self.halo2_M200 = None
        # self.halo2_conc = None

    def plot_shears(self,g1,g2,limit_mask=None,unit='arcmin',quiver_scale=1):
        

        ephi=0.5*np.arctan2(g2,g1)              

        if limit_mask==None:
            emag=np.sqrt(g1**2+g2**2)
        else:
            emag=np.sqrt(g1**2+g2**2) * limit_mask


        nuse=1
        # line_width=0.005* quiver_scale
        line_width=0.002

        if unit=='arcmin':
            pl.quiver(self.shear_u_arcmin[::nuse],self.shear_v_arcmin[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='b',label='original',scale=quiver_scale , width = line_width)  
            pl.xlim([min(self.shear_u_arcmin),max(self.shear_u_arcmin)])
            pl.ylim([min(self.shear_v_arcmin),max(self.shear_v_arcmin)])
        elif unit=='Mpc':
            pl.quiver(self.shear_u_mpc[::nuse],self.shear_v_mpc[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='b',label='original',scale=quiver_scale , width = line_width)  
            pl.xlim([min(self.shear_u_mpc),max(self.shear_u_mpc)])
            pl.ylim([min(self.shear_v_mpc),max(self.shear_v_mpc)])

        pl.axis('equal')
        pl.xlabel(unit)
        pl.ylabel(unit)

    def plot_shears_mag(self,g1,g2):

        emag=np.sqrt(g1**2+g2**2)

        
        pl.scatter(self.shear_u_arcmin,self.shear_v_arcmin,50,c=emag)
        pl.colorbar()
        pl.xlim([min(self.shear_u_arcmin),max(self.shear_u_arcmin)])
        pl.ylim([min(self.shear_v_arcmin),max(self.shear_v_arcmin)])
        pl.axis('equal')

    def plot_residual_whisker(self,model_g1,model_g2,limit_mask=None):

        res1  = (self.shear_g1 - model_g1) 
        res2  = (self.shear_g2 - model_g2) 
        
        pl.figure()
        pl.subplot(3,1,1)
        self.plot_shears(self.shear_g1,self.shear_g2,limit_mask)
        pl.subplot(3,1,2)
        self.plot_shears(model_g1,model_g2,limit_mask)
        pl.subplot(3,1,3)
        self.plot_shears(res1 , res2,limit_mask)


    def plot_residual_g1g2(self,model_g1,model_g2,limit_mask=None):

        res1  = (self.shear_g1 - model_g1) 
        res2  = (self.shear_g2 - model_g2) 
        scatter_size = 5

        maxg = max( [ max(abs(self.shear_g1.flatten())) ,  max(abs(self.shear_g2.flatten())) , max(abs(model_g1.flatten())) , max(abs(model_g1.flatten()))   ])

        pl.figure()

        pl.subplot(3,2,1)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, self.shear_g1 , lw = 0 , vmax=maxg , vmin=-maxg)
        pl.colorbar()
        pl.axis('equal')
        pl.subplot(3,2,3)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, model_g1 , lw = 0 , vmax=maxg , vmin=-maxg)
        # pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, model_g1 , lw = 0 )
        pl.colorbar()
        pl.axis('equal')
        pl.subplot(3,2,5)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, res1 , lw = 0 , vmax=maxg , vmin=-maxg)
        pl.colorbar()
        pl.axis('equal')

        pl.subplot(3,2,2)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, self.shear_g2 , lw = 0 , vmax=maxg , vmin=-maxg)
        pl.colorbar()
        pl.axis('equal')
        pl.subplot(3,2,4)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, model_g2 , lw = 0 , vmax=maxg , vmin=-maxg)
        # pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, model_g2 , lw = 0 )
        pl.colorbar()
        pl.axis('equal')
        pl.subplot(3,2,6)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, res2 , lw = 0 , vmax=maxg , vmin=-maxg)
        pl.colorbar()
        pl.axis('equal')
        
    def plot_model(self,p):

        model_g1 , model_g2, limit_mask , _ , _ = self.draw_model(p)
        self.plot_residual(model_g1 , model_g2, limit_mask)        

    def get_halo_signal(self):

        nh1 = nfw.NfwHalo()
        nh1.M_200=self.halo1_M200
        nh1.concentr=self.halo1_conc
        nh1.z_cluster= self.halo1_z
        nh1.theta_cx = self.halo1_u_arcmin
        nh1.theta_cy = self.halo1_v_arcmin 

        nh2 = nfw.NfwHalo()
        nh2.M_200=self.halo2_M200
        nh2.concentr=self.halo2_conc
        nh2.z_cluster= self.halo2_z
        nh2.theta_cx = self.halo2_u_arcmin
        nh2.theta_cy = self.halo2_v_arcmin 

        log.debug('got signal for halo1 u=%5.2f v=%5.2f' , nh1.theta_cx, nh1.theta_cy)
        log.debug('got signal for halo2 u=%5.2f v=%5.2f' , nh2.theta_cx, nh2.theta_cy)

        h1g1 , h1g2  = nh1.get_shears_with_pz(self.shear_u_arcmin , self.shear_v_arcmin , self.grid_z_centers , self.prob_z, redshift_offset)
        h2g1 , h2g2  = nh2.get_shears_with_pz(self.shear_u_arcmin , self.shear_v_arcmin , self.grid_z_centers , self.prob_z, redshift_offset)


        self.halos_g1 = h1g1 + h2g1
        self.halos_g2 = h1g2 + h2g2
        
    def get_concentr(self,M,z):

        # Duffy et al 2008 from King and Mead 2011
        # concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cosmology.cospars.h)**(-0.081)
        concentr = 5.72/(1.+z)**0.71 * (np.abs(M) / 1e14)**(-0.081)

        return concentr


    def draw_model(self,params):
        """
        params[0] filament_kappa
        params[1] filament_radius [Mpc]
        """

        pair_z = np.mean([self.halo1_z, self.halo2_z])

        halo1_M200 = params[2] * 1e14
        halo2_M200 = params[3] * 1e14

        self.nh1.M_200= halo1_M200
        self.nh1.update()
        self.nh1.R_200 = self.nh1.r_s*self.nh1.concentr

        self.nh2.M_200= halo2_M200
        self.nh2.update()
        self.nh2.R_200 = self.nh2.r_s*self.nh2.concentr      

        if self.kappa_is_K == 'DS': # model where kappa is dependent on halo mass
            warnings.warn('using DS')
            h1_r200_arcmin = self.nh1.R_200/cosmology.get_ang_diam_dist(self.halo1_z)/np.pi*180*60
            h2_r200_arcmin = self.nh1.R_200/cosmology.get_ang_diam_dist(self.halo1_z)/np.pi*180*60
            theta1_x=self.nh1.theta_cx+h1_r200_arcmin
            theta2_x=self.nh1.theta_cx+h1_r200_arcmin
            h1g1 , h1g2 , h1_DeltaSigma , h1_Sigma_crit, h1_kappa  = self.nh1.get_shears_with_pz_fast(np.array([theta1_x,theta1_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            h2g1 , h2g2 , h2_DeltaSigma , h2_Sigma_crit, h2_kappa  = self.nh1.get_shears_with_pz_fast(np.array([theta2_x,theta2_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            DeltaSigma_at_R200 = (np.abs(h1_DeltaSigma[0])+np.abs(h2_DeltaSigma[0]))/2.
            filament_kappa0 = params[0]*DeltaSigma_at_R200 / 1e14
            filament_radius = params[1]*(self.nh1.R_200+self.nh1.R_200)/2.
            # filament_radius = params[1]
            if self.n_model_evals % 1000==0:
                log.info('p[0]=%5.4f p[1]=%5.4f p[2]=%5.4f p[3]=%5.4f kappa0=%5.4e radius=%5.4f r200_1=%5.2f r200_2=%5.2f m200_1=%5.2e m200_2=%5.2e' % ( params[0],params[1],params[2],params[3],filament_kappa0,filament_radius,self.nh1.R_200,self.nh2.R_200,self.nh1.M_200,self.nh2.M_200 ))

        elif self.kappa_is_K == 'DSmean': # model where kappa is dependent on halo mass
            warnings.warn('using DSmean')
            h1_r200_arcmin = self.nh1.R_200/cosmology.get_ang_diam_dist(self.halo1_z)/np.pi*180*60
            h2_r200_arcmin = self.nh2.R_200/cosmology.get_ang_diam_dist(self.halo2_z)/np.pi*180*60
            theta1_x=self.nh1.theta_cx+h1_r200_arcmin
            theta2_x=self.nh2.theta_cx+h2_r200_arcmin
            h1g1 , h1g2 , h1_DeltaSigma , h1_Sigma_crit, h1_kappa  = self.nh1.get_shears_with_pz_fast(np.array([theta1_x,theta1_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            h2g1 , h2g2 , h2_DeltaSigma , h2_Sigma_crit, h2_kappa  = self.nh2.get_shears_with_pz_fast(np.array([theta2_x,theta2_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            DeltaSigma_at_R200 = (np.abs(h1_DeltaSigma[0])+np.abs(h2_DeltaSigma[0]))/2.
            filament_kappa0 = params[0]*DeltaSigma_at_R200 / 1e14
            filament_radius = params[1]*(self.nh1.R_200+self.nh2.R_200)/2.
            # filament_radius = params[1]
            if self.n_model_evals % 1000==log:
                log.info('p[0]=%5.4f p[1]=%5.4f p[2]=%5.4f p[3]=%5.4f kappa0=%5.4e radius=%5.4f r200_1=%5.2f r200_2=%5.2f m200_1=%5.2e m200_2=%5.2e' % ( params[0],params[1],params[2],params[3],filament_kappa0,filament_radius,self.nh1.R_200,self.nh2.R_200,self.nh1.M_200,self.nh2.M_200 ))

        elif self.kappa_is_K == 'DS-freeR': # model where kappa is dependent on halo mass
            warnings.warn('using DS-freeR')
            h1_r200_arcmin = self.nh1.R_200/cosmology.get_ang_diam_dist(self.halo1_z)/np.pi*180*60
            h2_r200_arcmin = self.nh1.R_200/cosmology.get_ang_diam_dist(self.halo1_z)/np.pi*180*60
            theta1_x=self.nh1.theta_cx+h1_r200_arcmin
            theta2_x=self.nh1.theta_cx+h1_r200_arcmin
            h1g1 , h1g2 , h1_DeltaSigma , h1_Sigma_crit, h1_kappa  = self.nh1.get_shears_with_pz_fast(np.array([theta1_x,theta1_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            h2g1 , h2g2 , h2_DeltaSigma , h2_Sigma_crit, h2_kappa  = self.nh1.get_shears_with_pz_fast(np.array([theta2_x,theta2_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            DeltaSigma_at_R200 = (np.abs(h1_DeltaSigma[0])+np.abs(h2_DeltaSigma[0]))/2.
            filament_kappa0 = params[0]*DeltaSigma_at_R200 / 1e14
            filament_radius = params[1]
            # filament_radius = params[1]
            if self.n_model_evals % 1000==0:
                log.info('p[0]=%5.4f p[1]=%5.4f p[2]=%5.4f p[3]=%5.4f kappa0=%5.4e radius=%5.4f r200_1=%5.2f r200_2=%5.2f m200_1=%5.2e m200_2=%5.2e' % ( params[0],params[1],params[2],params[3],filament_kappa0,filament_radius,self.nh1.R_200,self.nh2.R_200,self.nh1.M_200,self.nh2.M_200 ))

        if self.kappa_is_K == 'DSmean-freeR': # model where kappa is dependent on halo mass
            warnings.warn('using DSmean-freeR')
            h1_r200_arcmin = self.nh1.R_200/cosmology.get_ang_diam_dist(self.halo1_z)/np.pi*180*60
            h2_r200_arcmin = self.nh2.R_200/cosmology.get_ang_diam_dist(self.halo2_z)/np.pi*180*60
            theta1_x=self.nh1.theta_cx+h1_r200_arcmin
            theta2_x=self.nh2.theta_cx+h2_r200_arcmin
            h1g1 , h1g2 , h1_DeltaSigma , h1_Sigma_crit, h1_kappa  = self.nh1.get_shears_with_pz_fast(np.array([theta1_x,theta1_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            h2g1 , h2g2 , h2_DeltaSigma , h2_Sigma_crit, h2_kappa  = self.nh2.get_shears_with_pz_fast(np.array([theta2_x,theta2_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            DeltaSigma_at_R200 = (np.abs(h1_DeltaSigma[0])+np.abs(h2_DeltaSigma[0]))/2.
            filament_kappa0 = params[0]*DeltaSigma_at_R200 / 1e14
            filament_radius = params[1]
            # filament_radius = params[1]
            if self.n_model_evals % 1000==0:
                log.info('p[0]=%5.4f p[1]=%5.4f p[2]=%5.4f p[3]=%5.4f kappa0=%5.4e radius=%5.4f r200_1=%5.2f r200_2=%5.2f m200_1=%5.2e m200_2=%5.2e' % ( params[0],params[1],params[2],params[3],filament_kappa0,filament_radius,self.nh1.R_200,self.nh2.R_200,self.nh1.M_200,self.nh2.M_200 ))

        elif self.kappa_is_K == 'shear' : # model where kappa is dependent on halo mass
            h1_r200_arcmin = self.nh1.R_200/cosmology.get_ang_diam_dist(self.halo1_z)/np.pi*180*60
            h2_r200_arcmin = self.nh2.R_200/cosmology.get_ang_diam_dist(self.halo2_z)/np.pi*180*60
            theta1_x=self.nh1.theta_cx+h1_r200_arcmin
            theta2_x=self.nh2.theta_cx+h2_r200_arcmin
            h1g1 , h1g2 , h1_DeltaSigma , h1_Sigma_crit, h1_kappa  = self.nh1.get_shears_with_pz_fast(np.array([theta1_x,theta1_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            h2g1 , h2g2 , h2_DeltaSigma , h2_Sigma_crit, h2_kappa  = self.nh2.get_shears_with_pz_fast(np.array([theta2_x,theta2_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            DeltaSigma_at_R200 = (np.abs(h1g1[0]*h1_Sigma_crit)+np.abs(h2g1[0]*h2_Sigma_crit))/2.
            filament_kappa0 = params[0]*DeltaSigma_at_R200 / 1e14
            filament_radius = params[1]*(self.nh1.R_200+self.nh2.R_200)/2.
            if self.n_model_evals % 1000==0:
                log.info('p[0]=%5.4f p[1]=%5.4f p[2]=%5.4f p[3]=%5.4f kappa0=%5.4e radius=%5.4f r200_1=%5.2f r200_2=%5.2f m200_1=%5.2e m200_2=%5.2e' % ( params[0],params[1],params[2],params[3],filament_kappa0,filament_radius,self.nh1.R_200,self.nh2.R_200,self.nh1.M_200,self.nh2.M_200 ))

        elif self.kappa_is_K == 'null' : # model where kappa is dependent on halo mass
            h1_r200_arcmin = self.nh1.R_200/cosmology.get_ang_diam_dist(self.halo1_z)/np.pi*180*60
            h2_r200_arcmin = self.nh1.R_200/cosmology.get_ang_diam_dist(self.halo1_z)/np.pi*180*60
            theta1_x=self.nh1.theta_cx+h1_r200_arcmin
            theta2_x=self.nh1.theta_cx+h1_r200_arcmin
            h1g1 , h1g2 , h1_DeltaSigma , h1_Sigma_crit, h1_kappa  = self.nh1.get_shears_with_pz_fast(np.array([theta1_x,theta1_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            h2g1 , h2g2 , h2_DeltaSigma , h2_Sigma_crit, h2_kappa  = self.nh1.get_shears_with_pz_fast(np.array([theta2_x,theta2_x]) , np.array([0,0]) , self.grid_z_centers , self.prob_z, redshift_offset)
            DeltaSigma_at_R200 = (np.abs(h1g1[0]*h1_Sigma_crit)+np.abs(h2g1[0]*h2_Sigma_crit))/2.
            filament_kappa0 = params[0]*DeltaSigma_at_R200 / 1e14
            filament_radius = params[1]*(self.nh1.R_200+self.nh1.R_200)/2.
            # remove the halos signal completely for random points null test
            self.nh1.M_200= 1e-8
            self.nh2.M_200= 1e-8
            self.nh1.concentr = self.get_concentr(halo1_M200,self.halo1_z)
            self.nh2.concentr = self.get_concentr(halo2_M200,self.halo2_z)
            if self.n_model_evals % 10==0:
                log.info('p[0]=%5.4f p[1]=%5.4f p[2]=%5.4f p[3]=%5.4f kappa0=%5.4e radius=%5.4f r200_1=%5.2f r200_2=%5.2f m200_1=%5.2e m200_2=%5.2e' % ( params[0],params[1],params[2],params[3],filament_kappa0,filament_radius,self.nh1.R_200,self.nh2.R_200,self.nh1.M_200,self.nh2.M_200 ))

        else:   # standard model
            filament_kappa0 = params[0]
            filament_radius = params[1]
            halo2_M200 = params[3] * 1e14
            halo1_M200 = params[2] * 1e14

        if self.use_boost:
            filament_kappa0 *= self.boost

        filament_u1_mpc = self.halo1_u_mpc - self.R_start #*self.nh1.R_200
        filament_u2_mpc = self.halo2_u_mpc + self.R_start #*self.nh2.R_200

        h1g1 , h1g2 , h1_Delta_Sigma, h1_Sigma_crit, h1_kappa = self.nh1.get_shears_with_pz_fast(self.shear_u_arcmin , self.shear_v_arcmin , self.grid_z_centers , self.prob_z, redshift_offset)
        h2g1 , h2g2 , h2_Delta_Sigma, h2_Sigma_crit, h2_kappa  = self.nh2.get_shears_with_pz_fast(self.shear_u_arcmin , self.shear_v_arcmin , self.grid_z_centers , self.prob_z, redshift_offset)
        fg1  , fg2  , f_Delta_Sigma, f_Sigma_crit, f_kappa = self.filam.filament_model_with_pz(shear_u_mpc=self.shear_u_mpc, shear_v_mpc=self.shear_v_mpc , u1_mpc=filament_u1_mpc , u2_mpc=filament_u2_mpc ,  kappa0=filament_kappa0 ,  radius_mpc=filament_radius ,  pair_z=pair_z ,  grid_z_centers=self.grid_z_centers , prob_z=self.prob_z)


        model_g1 = h1g1 + h2g1 + fg1
        model_g2 = h1g2 + h2g2 + fg2
        model_kappa = h1_kappa + h2_kappa + f_kappa
        model_DeltaSigma = h1_Delta_Sigma + h2_Delta_Sigma + f_Delta_Sigma

        limit_mask = np.abs(model_g1 + 1j*model_g2) < weak_limit

        self.n_model_evals +=1

        return  model_g1 , model_g2 , limit_mask , model_DeltaSigma, model_kappa

    def log_posterior(self,theta):


        model_g1 , model_g2, limit_mask , _ , _ = self.draw_model(theta)


        likelihood = self.log_likelihood(model_g1,model_g2,limit_mask)
        prior = self.log_prior(theta)
        if not np.isfinite(prior):
            posterior = -np.inf
        else:
            # use no info from prior for now
            posterior = likelihood + prior 

        if log.level == logging.DEBUG:
            n_progress = 10
        elif log.level == logging.INFO:
            n_progress = 1000
        if self.n_model_evals % n_progress == 0:

            log.info('%7d post=% 2.8e like=% 2.8e prior=% 2.4e kappa0=% 6.3f radius=% 6.3f h1M200=% 5.2e h1M200=% 5.2e' % (self.n_model_evals,posterior,likelihood,prior,theta[0],theta[1],theta[2],theta[3]))

        if np.isnan(posterior):
            import pdb; pdb.set_trace()

        if self.save_all_models:

            self.plot_residual_whisker(model_g1,model_g2,limit_mask)

            pl.rcParams['font.size'] = 4
            pl.suptitle('model post=% 10.8e kappa0=%5.2e radius=%2.4f h1M200=% 5.2e h1M200=% 5.2e' % (posterior,theta[0],theta[1],10.**theta[2],10.**theta[3])) 
            filename_fig = 'models/res1.%04d.pdf' % self.n_model_evals
            pl.savefig(filename_fig,dpi=100)
            log.debug('saved %s' % filename_fig)
            pl.close()

            self.plot_residual_g1g2(model_g1,model_g2,limit_mask)

            pl.suptitle('model post=% 10.8e kappa0=%5.2e radius=%2.4f h1M200=% 5.2e h1M200=% 5.2e' % (posterior,theta[0],theta[1],10.**theta[2],10.**theta[3]) )
            filename_fig = 'models/res2.%04d.pdf' % self.n_model_evals
            pl.savefig(filename_fig,dpi=100)
            log.debug('saved %s' % filename_fig)
            pl.close()


        return posterior

    def log_prior(self,theta):
        
        kappa0 = theta[0]
        radius = theta[1]
        h1M200 = theta[2]
        h2M200 = theta[3]

        # prob = -0.5 * ( (log10_M200 - self.gaussian_prior_theta[0]['mean'])/self.gaussian_prior_theta[0]['std'] )**2  - np.log(np.sqrt(2*np.pi))
        prob=1e-20 # small number so that the prior doesn't matter

        if ( self.parameters[0]['box']['min'] <= kappa0 <= self.parameters[0]['box']['max'] ):
            if ( self.parameters[1]['box']['min'] <= radius <= self.parameters[1]['box']['max'] ):
                if ( self.parameters[2]['box']['min'] <= h1M200 <= self.parameters[2]['box']['max'] ):
                    if ( self.parameters[3]['box']['min'] <= h2M200 <= self.parameters[3]['box']['max'] ):
                        return prob
        return -np.inf
       
    def log_likelihood(self,model_g1,model_g2,limit_mask):

        res1_sq = ((model_g1 - self.shear_g1)**2) * self.inv_sq_sigma_g #* limit_mask
        res2_sq = ((model_g2 - self.shear_g2)**2) * self.inv_sq_sigma_g #* limit_mask
        # n_points = len(np.nonzero(limit_mask))

        # chi2 = -0.5 * ( np.sum( ((res1)/self.sigma_g) **2) + np.sum( ((res2)/self.sigma_g) **2) ) / n_points
        chi2 = -0.5 * ( np.sum( res1_sq )  + np.sum( res2_sq  ) )

        return chi2

    def null_log_likelihood(self,h1M200,h2M200):

        theta_null = [0,1,h1M200,h2M200]
        model_g1 , model_g2, limit_mask , _ , _ = self.draw_model(theta_null)
        null_log_like = self.log_likelihood(model_g1,model_g2,limit_mask)
        # null_log_post =  self.log_posterior(theta_null)
        
        # print null_log_like , null_log_post
        # pl.show()

        return null_log_like

    def run_mcmc(self):

        self.n_model_evals = 0

        self.pair_z  = (self.halo1_z + self.halo2_z) / 2.

        self.filam = filament.filament()
        self.filam.pair_z =self.pair_z
        self.filam.grid_z_centers = self.grid_z_centers
        self.filam.prob_z = self.prob_z
        self.filam.set_mean_inv_sigma_crit(self.filam.grid_z_centers,self.filam.prob_z,self.filam.pair_z)

        self.nh1 = nfw.NfwHalo()
        self.nh1.z_cluster= self.halo1_z
        self.nh1.theta_cx = self.halo1_u_arcmin
        self.nh1.theta_cy = self.halo1_v_arcmin 
        self.nh1.set_mean_inv_sigma_crit(self.grid_z_centers,self.prob_z,self.pair_z)

        self.nh2 = nfw.NfwHalo()
        self.nh2.z_cluster= self.halo2_z
        self.nh2.theta_cx = self.halo2_u_arcmin
        self.nh2.theta_cy = self.halo2_v_arcmin 
        self.nh2.set_mean_inv_sigma_crit(self.grid_z_centers,self.prob_z,self.pair_z)

        self.sampler = emcee.EnsembleSampler(nwalkers=self.n_walkers, dim=self.n_dim , lnpostfn=self.log_posterior)

        grid_kappa0 = np.linspace(self.parameters[0]['box']['min'],self.parameters[0]['box']['max'], self.parameters[0]['n_grid'])
        grid_radius = np.linspace(self.parameters[1]['box']['min'],self.parameters[1]['box']['max'], self.parameters[1]['n_grid'])
        grid_h1M200 = np.linspace(self.parameters[2]['box']['min'],self.parameters[2]['box']['max'], self.parameters[2]['n_grid'])
        grid_h2M200 = np.linspace(self.parameters[3]['box']['min'],self.parameters[3]['box']['max'], self.parameters[3]['n_grid'])
        n_total = len(grid_kappa0)* len(grid_radius)* len(grid_h1M200)* len(grid_h2M200)
        log_post = np.zeros([len(grid_kappa0), len(grid_radius) , len(grid_h1M200) , len(grid_h2M200)] )
        X1,X2,X3,X4 = np.meshgrid(grid_kappa0,grid_radius,grid_h1M200,grid_h2M200,indexing='ij')
        
        # get walkers
        theta0 = []
        for iw in range(self.n_walkers):
            start = [None]*self.n_dim
            for ip in range(self.n_dim):
                start[ip]  = np.random.uniform( low= -self.parameters[ip]['box']['max'] , high=self.parameters[ip]['box']['max'] ) 

            start = np.array(start)
            theta0.append( start )
        theta0 = np.array(theta0)
                 
        # run mcmc
        self.sampler.run_mcmc(theta0, self.n_samples)

        # cut burnin
        N_BURNIN = 100
        if N_BURNIN < len(self.sampler.flatchain):
            chain = self.sampler.chain[:, N_BURNIN:, :].reshape((-1, self.n_dim))
            chain_lnprob = self.sampler.lnprobability[:,N_BURNIN:].reshape(-1)
        else:
            chain = self.sampler.flatchain
            chain_lnprob = self.sampler.flatlnprobability

        diff=np.diff(chain,axis=0)
        norm_diff = np.linalg.norm(diff,axis=1)
        norm_diff=np.insert(norm_diff,0,0)
        select = norm_diff>1e-5
        chain = chain[select,:]
        chain_lnprob = chain_lnprob[select]

        # from samples to grid
        # from scipy.stats.kde import gaussian_kde
        # kde_est = gaussian_kde(chain.T)
        # grid_flat=np.concatenate([X1.flatten()[:,None],X2.flatten()[:,None],X3.flatten()[:,None],X4.flatten()[:,None]],axis=1)
        # kde_prob = kde_est(grid_flat.T)
        # log.info('params KDE bandwidth=%2.3f normalisation=%f', kde_est.factor , np.sum(kde_prob))
        # log_post = np.log(kde_prob)
        # select = np.isnan(log_post) | np.isinf(log_post)
        # log_post[select] = log_post[~select].min()
        # dims = [self.parameters[i]['n_grid'] for i in range(self.n_dim)]
        # log_post = np.reshape(log_post,dims)

        # save

        # get 2d margs
        from scipy.stats.kde import gaussian_kde
        marginals = np.zeros([self.n_dim,self.n_dim,self.n_mcmc_grid,self.n_mcmc_grid])

        for i1 in range(0,self.n_dim):
            for i2 in range(i1+1,self.n_dim):
                kde_est = gaussian_kde(chain[:,[i1,i2]].T)
                grid1=np.linspace(self.parameters[i1]['box']['min'],self.parameters[0]['box']['max'], self.n_mcmc_grid)
                grid2=np.linspace(self.parameters[i2]['box']['min'],self.parameters[0]['box']['max'], self.n_mcmc_grid)
                G1,G2=np.meshgrid(grid1,grid2,indexing='ij')
                grid_flat=np.concatenate([G1.flatten()[:,None],G2.flatten()[:,None]],axis=1)
                kde_prob = kde_est(grid_flat.T)
                log.info('params KDE bandwidth=%2.3f normalisation=%f', kde_est.factor , np.sum(kde_prob))
                log_post = np.log(kde_prob)
                select = np.isnan(log_post) | np.isinf(log_post)
                log_post[select] = log_post[~select].min()
                dims = [self.n_mcmc_grid,self.n_mcmc_grid]
                log_post = np.reshape(log_post,dims)
                marginals[i1,i2,:,:] = log_post

        grids = [X1,X2,X3,X4]
        params =  [grid_kappa0, grid_radius , grid_h1M200 , grid_h2M200]
        return log_post , params , grids , chain, chain_lnprob, marginals

    def run_gridsearch(self):
        
        self.n_model_evals = 0

        self.pair_z  = (self.halo1_z + self.halo2_z) / 2.

        self.filam = filament.filament()
        self.filam.pair_z =self.pair_z
        self.filam.grid_z_centers = self.grid_z_centers
        self.filam.prob_z = self.prob_z
        self.filam.set_mean_inv_sigma_crit(self.filam.grid_z_centers,self.filam.prob_z,self.filam.pair_z)

        self.nh1 = nfw.NfwHalo()
        self.nh1.z_cluster= self.halo1_z
        self.nh1.theta_cx = self.halo1_u_arcmin
        self.nh1.theta_cy = self.halo1_v_arcmin 
        self.nh1.set_mean_inv_sigma_crit(self.grid_z_centers,self.prob_z,self.pair_z)

        self.nh2 = nfw.NfwHalo()
        self.nh2.z_cluster= self.halo2_z
        self.nh2.theta_cx = self.halo2_u_arcmin
        self.nh2.theta_cy = self.halo2_v_arcmin 
        self.nh2.set_mean_inv_sigma_crit(self.grid_z_centers,self.prob_z,self.pair_z)

        self.n_model_evals = 0

        grid_kappa0 = np.linspace(self.parameters[0]['box']['min'],self.parameters[0]['box']['max'], self.parameters[0]['n_grid'])
        grid_radius = np.linspace(self.parameters[1]['box']['min'],self.parameters[1]['box']['max'], self.parameters[1]['n_grid'])
        grid_h1M200 = np.linspace(self.parameters[2]['box']['min'],self.parameters[2]['box']['max'], self.parameters[2]['n_grid'])
        grid_h2M200 = np.linspace(self.parameters[3]['box']['min'],self.parameters[3]['box']['max'], self.parameters[3]['n_grid'])

        n_total = len(grid_kappa0)* len(grid_radius)* len(grid_h1M200)* len(grid_h2M200)
        log_post = np.zeros([len(grid_kappa0), len(grid_radius) , len(grid_h1M200) , len(grid_h2M200)] )
        X1,X2,X3,X4 = np.meshgrid(grid_kappa0,grid_radius,grid_h1M200,grid_h2M200,indexing='ij')

        log.info('running gridsearch total %d grid points' % n_total)

        ia = 0
        for ik,vk in enumerate(grid_kappa0):
            for ir,vr in enumerate(grid_radius):
                for im1,vm1 in enumerate(grid_h1M200):
                    for im2,vm2 in enumerate(grid_h2M200):
                            
                            x1 = X1[ik,ir,im1,im2]    
                            x2 = X2[ik,ir,im1,im2]    
                            x3 = X3[ik,ir,im1,im2]    
                            x4 = X4[ik,ir,im1,im2]    

                            # log_post[ik,ir,im1,im2] = self.log_posterior([vk,vr,vm1,vm2])
                            # params[ia,:] = vk , vr , vm1 , vm2

                            log_post[ik,ir,im1,im2] = self.log_posterior([x1,x2,x3,x4])
                            ia+=1
                            # if ia % 1000 == 0 : log.info('gridsearch progress %d/%d models' , ia, n_total)

        # n_upsample = 100
        # grid_h1M200_hires=np.linspace(grid_h1M200.min(),grid_h1M200.max(),len(grid_h1M200)*n_upsample)
        # grid_h2M200_hires=np.linspace(grid_h2M200.min(),grid_h2M200.max(),len(grid_h2M200)*n_upsample)
        # log_prob_2D = np.zeros_like(log_post[:,:,0,0])
        # normalisation_const=log_post.max()
        # for i1 in range(len(log_post[:,0,0,0])):
        #     for i2 in range(len(log_post[0,:,0,0])):
        #         m200_2D = log_post[i1,i2,:,:]
        #         func_interp = scipy.interpolate.interp2d(grid_h1M200,grid_h2M200,m200_2D, kind='cubic')
        #         m200_2D_hires = func_interp(grid_h1M200_hires,grid_h2M200_hires)
        #         m200_2D_hires_prob=np.exp(m200_2D_hires-normalisation_const)
        #         log_prob_2D[i1,i2] = np.log(np.sum(m200_2D_hires_prob))

        #         import mathstools
        #         print grid_kappa0[i1], grid_radius[i2] , np.sum(np.exp(m200_2D-normalisation_const)) / float(len(m200_2D.flatten())) , np.sum(np.exp(m200_2D_hires-normalisation_const))/float(len(m200_2D_hires.flatten()))
                # pl.figure()
                # pl.subplot(1,2,1)
                # pl.imshow(mathstools.normalise(m200_2D),interpolation='nearest')
                # pl.subplot(1,2,2)
                # pl.imshow(mathstools.normalise(m200_2D_hires),interpolation='nearest')
                # pl.show()

        log_prob_2D = np.zeros(2)
        grids = [X1,X2,X3,X4]
        params =  [grid_kappa0, grid_radius , grid_h1M200 , grid_h2M200]
        return log_post , params , grids  , log_prob_2D


    def get_bcc_pz(self,filename_lenscat):

        if self.prob_z == None:

            # filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
            # filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'

            if 'fits' in filename_lenscat:
                lenscat = tabletools.loadTable(filename_lenscat)
                if 'z' in lenscat.dtype.names:
                    self.prob_z , _  = pl.histogram(lenscat['z'],bins=self.grid_z_edges,normed=True)
                elif 'z-phot' in lenscat.dtype.names:
                    self.prob_z , _  = pl.histogram(lenscat['z-phot'],bins=self.grid_z_edges,normed=True)

                if 'e1' in lenscat.dtype.names:

                    select = lenscat['star_flag'] == 0
                    lenscat = lenscat[select]
                    select = lenscat['fitclass'] == 0
                    lenscat = lenscat[select]
                    select = (lenscat['e1'] != 0.0) * (lenscat['e2'] != 0.0)
                    lenscat = lenscat[select]
                    self.sigma_ell = np.std(lenscat['e1']*lenscat['weight'],ddof=1)

            elif 'pp2' in filename_lenscat:

                pickle = tabletools.loadPickle(filename_lenscat)
                self.prob_z =  pickle['prob_z']
                self.grid_z_centers = pickle['bins_z']
                self.grid_z_edges = plotstools.get_bins_edges(self.grid_z_centers)
                

    def set_shear_sigma(self):

        # self.inv_sq_sigma_g = ( np.sqrt(self.shear_n_gals) / self.sigma_ell )**2
        self.inv_sq_sigma_g = self.shear_w

        # remove nans -- we shouldn't have nans in the data, but we appear to have
        select = np.isnan(self.shear_g1) | np.isnan(self.shear_g2)
        self.inv_sq_sigma_g[select] = 0
        self.shear_g1[select] = 0
        self.shear_g2[select] = 0
        n_nans = sum(np.isnan(self.shear_g1))
        log.info('found %d nan pixels' % n_nans)


    def get_grid_max(self,log_post,params):

        imax_index = np.unravel_index(log_post.argmax(), log_post.shape)
        vmax_post = log_post[imax_index]

        vmax_kappa0 = params[0][imax_index[0]]
        vmax_radius = params[1][imax_index[1]]
        vmax_h1M200 = params[2][imax_index[2]]
        vmax_h2M200 = params[3][imax_index[3]]

        vmax_params = [vmax_kappa0,vmax_radius,vmax_h1M200,vmax_h2M200]
        
        log.info('ML solution log_like=% 5.2e kappa0=% 5.2f radius=% 5.2f h1M200=% 5.2e h2M200=% 5.2e', vmax_post , vmax_kappa0 , vmax_radius , vmax_h1M200 , vmax_h2M200 )

        best_model_g1, best_model_g2, limit_mask , _ , _ = self.draw_model( vmax_params )

        return  vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_params

    def get_samples_max(self,log_post,params):

        vmax_post = log_post.max()
        imax_post = log_post.argmax()
        vmax_params = params[imax_post,:]
        vmax_kappa0 = vmax_params[0]
        vmax_radius = vmax_params[1]
        vmax_h1M200 = vmax_params[2]
        vmax_h2M200 = vmax_params[3]
        vmax_params = vmax_params[:]
        import pdb; pdb.set_trace()

        best_model_g1, best_model_g2, limit_mask , _ , _ = self.draw_model( vmax_params )

        log.info('ML solution log_like=% 5.2e kappa0=% 5.2f radius=% 5.2f h1M200=% 5.2e h2M200=% 5.2e', vmax_post , vmax_kappa0 , vmax_radius , 10.**vmax_h1M200 , 10.**vmax_h2M200 )

        return  vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_params

  
def test():

    filename_pairs = 'pairs_bcc.fits'
    filename_halo1 = 'pairs_bcc.halos1.fits'
    filename_halo2 = 'pairs_bcc.halos2.fits'
    filename_shears = 'shears_bcc_g.fits' 

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    sigma_g_add =  0.0001

    id_pair = 48
    shears_info = tabletools.loadTable(filename_shears,hdu=id_pair+1)

    fitobj = modelfit()
    fitobj.get_bcc_pz()
    fitobj.shear_u_arcmin =  shears_info['u_arcmin']
    fitobj.shear_v_arcmin =  shears_info['v_arcmin']
    fitobj.shear_u_mpc =  shears_info['u_mpc']
    fitobj.shear_v_mpc =  shears_info['v_mpc']
    fitobj.shear_g1 =  shears_info['g1'] + np.random.randn(len(shears_info['g1']))*sigma_g_add
    fitobj.shear_g2 =  shears_info['g2'] + np.random.randn(len(shears_info['g2']))*sigma_g_add
    fitobj.sigma_g =  np.std(shears_info['g2'],ddof=1)

    # fitobj.save_all_models = True

    log.info('using sigma_g=%2.5f' , fitobj.sigma_g)

    fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
    fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
    fitobj.halo1_u_mpc =  pairs_table['u1_mpc'][id_pair]
    fitobj.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair]
    fitobj.halo1_z =  pairs_table['z'][id_pair]

    fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
    fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
    fitobj.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair]
    fitobj.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair]
    fitobj.halo2_z =  pairs_table['z'][id_pair]

    fitobj.parameters[0]['box']['min'] = 0
    fitobj.parameters[0]['box']['max'] = 1
    fitobj.parameters[1]['box']['min'] = 1
    fitobj.parameters[1]['box']['max'] = 10
    fitobj.parameters[2]['box']['min'] = 14
    fitobj.parameters[2]['box']['max'] = 15
    fitobj.parameters[3]['box']['min'] = 14
    fitobj.parameters[3]['box']['max'] = 15

    print 'halo1 m200' , halo1_table['m200'][id_pair]
    print 'halo2 m200' , halo2_table['m200'][id_pair]

    # import pdb; pdb.set_trace()

    # fitobj.plot_shears_mag(fitobj.shear_g1,fitobj.shear_g2)
    # pl.show()
    fitobj.save_all_models=False
    log.info('running grid search')
    n_grid=10
    log_post , params, grids = fitobj.run_gridsearch(n_grid=n_grid)

    vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_params = fitobj.get_grid_max(log_post , params)

def self_fit():

    filename_pairs = 'pairs_bcc.fits'
    filename_halo1 = 'pairs_bcc.halos1.fits'
    filename_halo2 = 'pairs_bcc.halos2.fits'
    filename_shears = 'shears_bcc_g.fits' 

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    sigma_g_add =  0.1

    id_pair = 48
    shears_info = tabletools.loadTable(filename_shears,hdu=id_pair+1)

    fitobj = modelfit()
    fitobj.get_bcc_pz('aardvarkv1.0_des_lenscat_s2n10.351.fit')

    fitobj.halo1_z = 0.2
    fitobj.halo2_z = 0.2
    fitobj.halo1_u_arcmin = 20
    fitobj.halo1_v_arcmin = 0
    fitobj.halo2_u_arcmin = -20
    fitobj.halo2_v_arcmin = 0
    fitobj.shear_v_arcmin =  shears_info['v_arcmin']
    fitobj.shear_u_mpc =  shears_info['u_mpc']
    fitobj.shear_v_mpc =  shears_info['v_mpc']

    fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
    fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
    fitobj.halo1_u_mpc =  pairs_table['u1_mpc'][id_pair]
    fitobj.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair]
    fitobj.halo1_z =  pairs_table['z'][id_pair]

    fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
    fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
    fitobj.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair]
    fitobj.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair]
    fitobj.halo2_z =  pairs_table['z'][id_pair]

    fitobj.pair_z  = (fitobj.halo1_z + fitobj.halo2_z) / 2.

    fitobj.filam = filament.filament()
    fitobj.filam.pair_z =fitobj.pair_z
    fitobj.filam.grid_z_centers = fitobj.grid_z_centers
    fitobj.filam.prob_z = fitobj.prob_z
    fitobj.filam.set_mean_inv_sigma_crit(fitobj.filam.grid_z_centers,fitobj.filam.prob_z,fitobj.filam.pair_z)

    fitobj.nh1 = nfw.NfwHalo()
    fitobj.nh1.z_cluster= fitobj.halo1_z
    fitobj.nh1.theta_cx = fitobj.halo1_u_arcmin
    fitobj.nh1.theta_cy = fitobj.halo1_v_arcmin 
    fitobj.nh1.set_mean_inv_sigma_crit(fitobj.grid_z_centers,fitobj.prob_z,fitobj.pair_z)

    fitobj.nh2 = nfw.NfwHalo()
    fitobj.nh2.z_cluster= fitobj.halo2_z
    fitobj.nh2.theta_cx = fitobj.halo2_u_arcmin
    fitobj.nh2.theta_cy = fitobj.halo2_v_arcmin 
    fitobj.nh2.set_mean_inv_sigma_crit(fitobj.grid_z_centers,fitobj.prob_z,fitobj.pair_z)

    fitobj.shear_u_arcmin =  shears_info['u_arcmin']

    shear_model_g1, shear_model_g2, limit_mask , _ , _ = fitobj.draw_model([0., 2., 14.5, 14.5])
    fitobj.plot_shears(shear_model_g1, shear_model_g2,quiver_scale=0.5)
    pl.show()

    fitobj.shear_g1 =  shear_model_g1 + np.random.randn(len(shears_info['g1']))*sigma_g_add
    fitobj.shear_g2 =  shear_model_g2 + np.random.randn(len(shears_info['g2']))*sigma_g_add
    fitobj.sigma_g =  np.std(shear_model_g2,ddof=1)
    fitobj.inv_sq_sigma_g = 1./sigma_g_add**2
    log.info('using sigma_g=%2.5f' , fitobj.sigma_g)

    fitobj.parameters[0]['box']['min'] = 0
    fitobj.parameters[0]['box']['max'] = 1

    fitobj.parameters[1]['box']['min'] = 1
    fitobj.parameters[1]['box']['max'] = 10

    fitobj.parameters[2]['box']['min'] = 14
    fitobj.parameters[2]['box']['max'] = 15

    fitobj.parameters[3]['box']['min'] = 14
    fitobj.parameters[3]['box']['max'] = 15

    print 'halo1 m200' , halo1_table['m200'][id_pair]
    print 'halo2 m200' , halo2_table['m200'][id_pair]

    # import pdb; pdb.set_trace()
    # fitobj.plot_shears_mag(fitobj.shear_g1,fitobj.shear_g2)
    # pl.show()

    fitobj.save_all_models=False
    log.info('running grid search')
    fitobj.run_gridsearch()
       

    # fitobj.save_all_models=False
    # log.info('running mcmc search')
    # fitobj.n_walkers=10
    # fitobj.n_samples=2000
    # fitobj.run_mcmc()
    # params = fitobj.sampler.flatchain

    # plotstools.plot_dist(params)
    # pl.show()

    # vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_params = fitobj.get_grid_max(log_post , params)

def get_mock_data():

    filename_pairs =  config['filename_pairs']                                   # pairs_bcc.fits'
    filename_halo1 =  config['filename_pairs'].replace('.fits' , '.halos1.fits') # pairs_bcc.halos1.fits'
    filename_halo2 =  config['filename_pairs'].replace('.fits' , '.halos2.fits') # pairs_bcc.halos2.fits'
    filename_shears = config['filename_shears']                                  # args.filename_shears 

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    sigma_g_add =  0.

    id_pair = 2
    shears_info = tabletools.loadPickle(filename_shears,id_pair)

    fitobj = modelfit()
    fitobj.get_bcc_pz(config['filename_pz'])

    fitobj.shear_v_arcmin =  shears_info['v_arcmin']
    fitobj.shear_u_arcmin =  shears_info['u_arcmin']
    fitobj.shear_u_mpc =  shears_info['u_mpc']
    fitobj.shear_v_mpc =  shears_info['v_mpc']

    fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
    fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
    fitobj.halo1_u_mpc =  pairs_table['u1_mpc'][id_pair]
    fitobj.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair]
    fitobj.halo1_z =  pairs_table['z'][id_pair]

    fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
    fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
    fitobj.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair]
    fitobj.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair]
    fitobj.halo2_z =  pairs_table['z'][id_pair]

    fitobj.pair_z  = (fitobj.halo1_z + fitobj.halo2_z) / 2.

    fitobj.filam = filament.filament()
    fitobj.filam.pair_z =fitobj.pair_z
    fitobj.filam.grid_z_centers = fitobj.grid_z_centers
    fitobj.filam.prob_z = fitobj.prob_z
    fitobj.filam.set_mean_inv_sigma_crit(fitobj.filam.grid_z_centers,fitobj.filam.prob_z,fitobj.filam.pair_z)

    fitobj.nh1 = nfw.NfwHalo()
    fitobj.nh1.z_cluster= fitobj.halo1_z
    fitobj.nh1.theta_cx = fitobj.halo1_u_arcmin
    fitobj.nh1.theta_cy = fitobj.halo1_v_arcmin 
    fitobj.nh1.set_mean_inv_sigma_crit(fitobj.grid_z_centers,fitobj.prob_z,fitobj.pair_z)

    fitobj.nh2 = nfw.NfwHalo()
    fitobj.nh2.z_cluster= fitobj.halo2_z
    fitobj.nh2.theta_cx = fitobj.halo2_u_arcmin
    fitobj.nh2.theta_cy = fitobj.halo2_v_arcmin 
    fitobj.nh2.set_mean_inv_sigma_crit(fitobj.grid_z_centers,fitobj.prob_z,fitobj.pair_z)

    fitobj.shear_u_arcmin =  shears_info['u_arcmin']

    shear_model_g1, shear_model_g2, limit_mask = fitobj.draw_model([0.4, 0.5, 14., 14,])
    pl.figure()
    pl.scatter( fitobj.shear_u_mpc , fitobj.shear_v_mpc , c=shear_model_g1)   
    pl.colorbar()
    pl.figure()
    pl.scatter( fitobj.shear_u_mpc , fitobj.shear_v_mpc , c=shear_model_g2)
    pl.figure()
    fitobj.plot_shears(shear_model_g1,shear_model_g2,limit_mask,quiver_scale=2)
    pl.show()

    fitobj.shear_g1 =  shear_model_g1 + np.random.randn(len(shears_info['g1']))*sigma_g_add
    fitobj.shear_g2 =  shear_model_g2 + np.random.randn(len(shears_info['g2']))*sigma_g_add
    fitobj.sigma_g =  np.std(shear_model_g2,ddof=1)
    fitobj.inv_sq_sigma_g = 1./sigma_g_add**2
    log.info('using sigma_g=%2.5f' , fitobj.sigma_g)

def test_overlap():

    filename_pairs =  config['filename_pairs']                                   # pairs_bcc.fits'
    filename_halo1 =  config['filename_pairs'].replace('.fits' , '.halos1.fits') # pairs_bcc.halos1.fits'
    filename_halo2 =  config['filename_pairs'].replace('.fits' , '.halos2.fits') # pairs_bcc.halos2.fits'
    filename_shears = config['filename_shears']                                  # args.filename_shears 

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    sigma_g_add =  0.01

    id_pair = 2
    shears_info = tabletools.loadPickle(filename_shears,id_pair)

    fitobj = modelfit()
    fitobj.get_bcc_pz(config['filename_pz'])
    fitobj.kappa_is_K = config['kappa_is_K']
    fitobj.R_start = config['R_start']
    fitobj.Dlos = pairs_table[id_pair]['Dlos']        
    fitobj.Dtot = np.sqrt(pairs_table[id_pair]['Dxy']**2+pairs_table[id_pair]['Dlos']**2)
    fitobj.boost = fitobj.Dtot/pairs_table[id_pair]['Dxy']
    fitobj.use_boost = config['use_boost']

    fitobj.shear_v_arcmin =  shears_info['v_arcmin'][::4]
    fitobj.shear_u_arcmin =  shears_info['u_arcmin'][::4]
    fitobj.shear_u_mpc =  shears_info['u_mpc'][::4]
    fitobj.shear_v_mpc =  shears_info['v_mpc'][::4]

    fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
    fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
    fitobj.halo1_u_mpc =  pairs_table['u1_mpc'][id_pair]
    fitobj.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair]
    fitobj.halo1_z =  pairs_table['z'][id_pair]

    fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
    fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
    fitobj.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair]
    fitobj.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair]
    fitobj.halo2_z =  pairs_table['z'][id_pair]

    fitobj.pair_z  = (fitobj.halo1_z + fitobj.halo2_z) / 2.

    fitobj.filam = filament.filament()
    fitobj.filam.pair_z =fitobj.pair_z
    fitobj.filam.grid_z_centers = fitobj.grid_z_centers
    fitobj.filam.prob_z = fitobj.prob_z
    fitobj.filam.set_mean_inv_sigma_crit(fitobj.filam.grid_z_centers,fitobj.filam.prob_z,fitobj.filam.pair_z)

    fitobj.nh1 = nfw.NfwHalo()
    fitobj.nh1.z_cluster= fitobj.halo1_z
    fitobj.nh1.theta_cx = fitobj.halo1_u_arcmin
    fitobj.nh1.theta_cy = fitobj.halo1_v_arcmin 
    fitobj.nh1.set_mean_inv_sigma_crit(fitobj.grid_z_centers,fitobj.prob_z,fitobj.pair_z)

    fitobj.nh2 = nfw.NfwHalo()
    fitobj.nh2.z_cluster= fitobj.halo2_z
    fitobj.nh2.theta_cx = fitobj.halo2_u_arcmin + 10
    fitobj.nh2.theta_cy = fitobj.halo2_v_arcmin + 5     # add mis-centering
    fitobj.nh2.set_mean_inv_sigma_crit(fitobj.grid_z_centers,fitobj.prob_z,fitobj.pair_z)
    shear_model_g1, shear_model_g2, limit_mask , _ , _  = fitobj.draw_model([0.0, 0.5, 14., 14,])
    fitobj.nh2.theta_cy = fitobj.halo2_v_arcmin - 5     # remobe miscentering
    fitobj.nh2.theta_cx = fitobj.halo2_u_arcmin - 10

    # second fitobj
    fitobj2 = modelfit()
    fitobj2.get_bcc_pz(config['filename_pz'])
    fitobj2.use_boost = True
    fitobj2.kappa_is_K = config['kappa_is_K']
    fitobj2.R_start = config['R_start']
    fitobj2.Dlos = pairs_table[id_pair]['Dlos']        
    fitobj2.Dtot = np.sqrt(pairs_table[id_pair]['Dxy']**2+pairs_table[id_pair]['Dlos']**2)
    fitobj2.boost = fitobj.Dtot/pairs_table[id_pair]['Dxy']
    fitobj2.use_boost = config['use_boost']

    fitobj2.shear_v_arcmin =  shears_info['v_arcmin'][::4]
    fitobj2.shear_u_arcmin =  shears_info['u_arcmin'][::4]
    fitobj2.shear_u_mpc =  shears_info['u_mpc'][::4]
    fitobj2.shear_v_mpc =  shears_info['v_mpc'][::4]

    fitobj2.halo1_u_arcmin =  (pairs_table['u1_arcmin'][id_pair] + pairs_table['u2_arcmin'][id_pair])/2.
    fitobj2.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair] + 20 
    fitobj2.halo1_u_mpc =  (pairs_table['u1_mpc'][id_pair]  + pairs_table['u1_mpc'][id_pair])/2.
    fitobj2.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair] + 3.77
    fitobj2.halo1_z =  pairs_table['z'][id_pair]

    fitobj2.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
    fitobj2.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
    fitobj2.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair]
    fitobj2.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair]
    fitobj2.halo2_z =  pairs_table['z'][id_pair]

    fitobj2.pair_z  = (fitobj2.halo1_z + fitobj2.halo2_z) / 2.

    fitobj2.filam = filament.filament()
    fitobj2.filam.pair_z =fitobj2.pair_z
    fitobj2.filam.grid_z_centers = fitobj2.grid_z_centers
    fitobj2.filam.prob_z = fitobj2.prob_z
    fitobj2.filam.set_mean_inv_sigma_crit(fitobj2.filam.grid_z_centers,fitobj2.filam.prob_z,fitobj2.filam.pair_z)

    fitobj2.nh1 = nfw.NfwHalo()
    fitobj2.nh1.z_cluster= fitobj2.halo1_z
    fitobj2.nh1.theta_cx = fitobj2.halo1_u_arcmin
    fitobj2.nh1.theta_cy = fitobj2.halo1_v_arcmin 
    fitobj2.nh1.set_mean_inv_sigma_crit(fitobj2.grid_z_centers,fitobj2.prob_z,fitobj2.pair_z)

    fitobj2.nh2 = nfw.NfwHalo()
    fitobj2.nh2.z_cluster= fitobj2.halo2_z
    fitobj2.nh2.theta_cx = fitobj2.halo2_u_arcmin
    fitobj2.nh2.theta_cy = fitobj2.halo2_v_arcmin 
    fitobj2.nh2.set_mean_inv_sigma_crit(fitobj2.grid_z_centers,fitobj2.prob_z,fitobj2.pair_z)

    shear_model_g1_neighbour, shear_model_g2_neighbour, limit_mask , _ , _  = fitobj2.draw_model([0.0, 0.5, 10, 10,])


    pl.figure()
    pl.scatter( fitobj.shear_u_mpc , fitobj.shear_v_mpc , c=shear_model_g1+shear_model_g1_neighbour , lw=0)   
    pl.colorbar()
    pl.figure()
    pl.scatter( fitobj.shear_u_mpc , fitobj.shear_v_mpc , c=shear_model_g2+shear_model_g2_neighbour, lw=0)
    pl.figure()
    fitobj.plot_shears(shear_model_g1,shear_model_g2,limit_mask,quiver_scale=2)
    pl.show()

    fitobj.shear_g1 =  shear_model_g1 + np.random.randn(len(fitobj.shear_u_arcmin))*sigma_g_add
    fitobj.shear_g2 =  shear_model_g2 + np.random.randn(len(fitobj.shear_u_arcmin))*sigma_g_add
    fitobj.sigma_g =  np.std(shear_model_g2,ddof=1)
    fitobj.inv_sq_sigma_g = 1./sigma_g_add**2
    log.info('using sigma_g=%2.5f' , fitobj.sigma_g)

    fitobj.parameters[0]['box']['min'] = config['kappa0']['box']['min']
    fitobj.parameters[0]['box']['max'] = config['kappa0']['box']['max']
    fitobj.parameters[0]['n_grid'] = config['kappa0']['n_grid']

    fitobj.parameters[1]['box']['min'] = config['radius']['box']['min']
    fitobj.parameters[1]['box']['max'] = config['radius']['box']['max']
    fitobj.parameters[1]['n_grid'] = config['radius']['n_grid']
    
    fitobj.parameters[2]['box']['min'] = config['h1M200']['box']['min']
    fitobj.parameters[2]['box']['max'] = config['h1M200']['box']['max']
    fitobj.parameters[2]['n_grid'] = config['h1M200']['n_grid']
    
    fitobj.parameters[3]['box']['min'] = config['h2M200']['box']['min']
    fitobj.parameters[3]['box']['max'] = config['h2M200']['box']['max']
    fitobj.parameters[3]['n_grid'] = config['h2M200']['n_grid']

    log_post , params, grids = fitobj.run_gridsearch()

    import pdb; pdb.set_trace()
    grid_info = {}
    grid_info['grid_kappa0'] = grids[0]
    grid_info['grid_radius'] = grids[1]
    grid_info['grid_h1M200'] = grids[2]
    grid_info['grid_h2M200'] = grids[3]
    grid_info['post_kappa0'] = params[0]
    grid_info['post_radius'] = params[1]
    grid_info['post_h1M200'] = params[2]
    grid_info['post_h2M200'] = params[3]
    filename_results_grid = 'results/results.grid.' +   os.path.basename(filename_shears).replace('.fits','.pp2')
    tabletools.savePickle(filename_results_grid,grid_info)

    filename_results_prob = 'results/results.prob.%04d.%04d.' % (0, 1) +   os.path.basename(filename_shears).replace('.fits','.pp2')
    tabletools.savePickle(filename_results_prob,log_post.astype(np.float32),append=False)

if __name__=='__main__':

    description = 'filaments_model_2hf'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')

    global args

    args = parser.parse_args()
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    log.setLevel(logging_level)

    global config 
    config = yaml.load(open(args.filename_config))

    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    # get_mock_data()
    test_overlap()

    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

