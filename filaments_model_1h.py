import os, yaml, argparse, sys, logging , pyfits,  emcee, tabletools, cosmology, filaments_tools, nfw, plotstools, filament
import numpy as np
import pylab as pl
import warnings

warnings.simplefilter('once')

log = logging.getLogger("fil_model_1h") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

redshift_offset = 0.2
weak_limit = 0.2

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

        self.n_dim = 1
        self.parameters = [None]*self.n_dim
        self.parameters[0] = {}       
        self.parameters[0]['name'] = 'halo_M200' 
        self.parameters[0]['box'] = {'min' : 13, 'max': 15}

        self.shear_g1 = None
        self.shear_g2 = None
        self.shear_n_gals = None
        self.shear_u_arcmin = None
        self.shear_v_arcmin = None
        self.halo_u_arcmin = None
        self.halo_v_arcmin = None
        

    def plot_shears(self,g1,g2,limit_mask=None,unit='arcmin',quiver_scale=1):
        

        ephi=0.5*np.arctan2(g2,g1)              

        if limit_mask==None:
            emag=np.sqrt(g1**2+g2**2)
        else:
            emag=np.sqrt(g1**2+g2**2) * limit_mask


        nuse=1
        line_width=0.005* quiver_scale

        if unit=='arcmin':
            pl.quiver(self.shear_u_arcmin[::nuse],self.shear_v_arcmin[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale , width = line_width)  
            pl.xlim([min(self.shear_u_arcmin),max(self.shear_u_arcmin)])
            pl.ylim([min(self.shear_v_arcmin),max(self.shear_v_arcmin)])
        elif unit=='Mpc':
            pl.quiver(self.shear_u_mpc[::nuse],self.shear_v_mpc[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale , width = line_width)  
            pl.xlim([min(self.shear_u_mpc),max(self.shear_u_mpc)])
            pl.ylim([min(self.shear_v_mpc),max(self.shear_v_mpc)])

        pl.axis('equal')

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

        pl.figure()

        pl.subplot(3,2,1)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, self.shear_g1 , lw = 0 )
        pl.colorbar()
        pl.axis('equal')
        pl.subplot(3,2,3)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, model_g1 , lw = 0 )
        # pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, model_g1 , lw = 0 )
        pl.colorbar()
        pl.axis('equal')
        pl.subplot(3,2,5)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, res1 , lw = 0 )
        pl.colorbar()
        pl.axis('equal')

        pl.subplot(3,2,2)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, self.shear_g2 , lw = 0 )
        pl.colorbar()
        pl.axis('equal')
        pl.subplot(3,2,4)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, model_g2 , lw = 0 )
        # pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, model_g2 , lw = 0 )
        pl.colorbar()
        pl.axis('equal')
        pl.subplot(3,2,6)
        pl.scatter(self.shear_u_arcmin, self.shear_v_arcmin, scatter_size, res2 , lw = 0 )
        pl.colorbar()
        pl.axis('equal')
        
        
    def get_concentr(self,M,z):

        # Duffy et al 2008 from King and Mead 2011
        # concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cosmology.cospars.h)**(-0.081)
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14)**(-0.081)

        return concentr

    def draw_model(self,params):
        """
        params[0] filament_kappa
        params[1] filament_radius [Mpc]
        """

        self.n_model_evals +=1
        pair_z = np.mean([self.halo_z])
        halo_M200 = params[0]

        self.nh.M_200= halo_M200
        self.nh.concentr = self.get_concentr(halo_M200,self.halo_z)
        self.nh.R_200 = self.nh.r_s*self.nh.concentr

        model_g1 , model_g2  = self.nh.get_shears_with_pz_fast(self.shear_u_arcmin , self.shear_v_arcmin , self.grid_z_centers , self.prob_z, redshift_offset)

        limit_mask = np.abs(model_g1 + 1j*model_g2) < weak_limit

        return  model_g1 , model_g2 , limit_mask

    def log_posterior(self,theta):

        model_g1 , model_g2, limit_mask = self.draw_model(theta)

        likelihood = self.log_likelihood(model_g1,model_g2,limit_mask)
        prior = self.log_prior(theta)
        if not np.isfinite(prior):
            posterior = -np.inf
        else:
            # use no info from prior for now
            posterior = likelihood 

        if log.level == logging.DEBUG:
            n_progress = 10
        elif log.level == logging.INFO:
            n_progress = 1000
        if self.n_model_evals % n_progress == 0:

            log.info('%7d post=% 2.8e like=% 2.8e prior=% 2.4e M200=% 6.3f ' % (self.n_model_evals,posterior,likelihood,prior,theta[0]))

        if np.isnan(posterior):
            import pdb; pdb.set_trace()

        if self.save_all_models:

            self.plot_residual_g1g2(model_g1,model_g2,limit_mask)

            pl.suptitle('model post=% 10.8e M200=%5.2e' % (posterior,theta[0]) )
            filename_fig = 'models/res2.%04d.png' % self.n_model_evals
            pl.savefig(filename_fig)
            log.debug('saved %s' % filename_fig)
            pl.close()


        return posterior

    def log_prior(self,theta):
        
        kappa0 = theta[0]

        # prob = -0.5 * ( (log10_M200 - self.gaussian_prior_theta[0]['mean'])/self.gaussian_prior_theta[0]['std'] )**2  - np.log(np.sqrt(2*np.pi))
        prob=1e-10 # small number so that the prior doesn't matter

        if ( self.parameters[0]['box']['min'] <= kappa0 <= self.parameters[0]['box']['max'] ):
                        return prob

        return -np.inf
       
    def log_likelihood(self,model_g1,model_g2,limit_mask):

        select = (~np.isnan(model_g1)) & (~np.isnan(model_g2)) & (~np.isinf(model_g1)) & (~np.isinf(model_g2))

        res1_sq = ((model_g1[select] - self.shear_g1[select])**2) * self.inv_sq_sigma_g[select] #* limit_mask
        res2_sq = ((model_g2[select] - self.shear_g2[select])**2) * self.inv_sq_sigma_g[select] #* limit_mask
        # n_points = len(np.nonzero(limit_mask))

        # chi2 = -0.5 * ( np.sum( ((res1)/self.sigma_g) **2) + np.sum( ((res2)/self.sigma_g) **2) ) / n_points
        chi2 = -0.5 * ( np.sum( res1_sq )  + np.sum( res2_sq  ) )

        return chi2

    def null_log_likelihood(self,h1M200,h2M200):

        theta_null = [0,1,h1M200,h2M200]
        model_g1 , model_g2, limit_mask = self.draw_model(theta_null)
        null_log_like = self.log_likelihood(model_g1,model_g2,limit_mask)
        # null_log_post =  self.log_posterior(theta_null)
        
        # print null_log_like , null_log_post
        # pl.show()

        return null_log_like

    def run_gridsearch(self):
        
        self.n_model_evals = 0

        self.nh = nfw.NfwHalo()
        self.nh.z_cluster= self.halo_z
        self.nh.theta_cx = self.halo_u_arcmin
        self.nh.theta_cy = self.halo_v_arcmin 
        self.nh.set_mean_inv_sigma_crit(self.grid_z_centers,self.prob_z,self.halo_z)

        self.n_model_evals = 0

        grid_M200 = np.linspace(self.parameters[0]['box']['min'],self.parameters[0]['box']['max'], self.parameters[0]['n_grid'])
        n_total = len(grid_M200)
        log_post = np.zeros([len(grid_M200)])
        
        log.info('running gridsearch total %d grid points' % n_total)

        ia = 0
        for ik,vk in enumerate(grid_M200):
            
            log_post[ik] = self.log_posterior([grid_M200[ik]])
            ia+=1
                
        return log_post , grid_M200     


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

                pickle = tabletools.loadPickle(filename_lenscat,log=0)
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


 