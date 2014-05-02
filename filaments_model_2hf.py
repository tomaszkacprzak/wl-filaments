import os, yaml, argparse, sys, logging , pyfits,  emcee, tabletools, cosmology, filaments_tools, nfw, plotstools, filament
import numpy as np
import pylab as pl
import warnings

warnings.simplefilter('once')

log = logging.getLogger("fil_mod_2hf") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

redshift_offset = 0.2
weak_limit = 0.2

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

        model_g1 , model_g2, limit_mask = self.draw_model(p)
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
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14)**(-0.081)

        return concentr


    def draw_model(self,params):
        """
        params[0] filament_kappa
        params[1] filament_radius [Mpc]
        """

        self.n_model_evals +=1
        pair_z = np.mean([self.halo1_z, self.halo2_z])

        filament_kappa0 = params[0]
        filament_radius = params[1]
        halo2_M200 = 10.**params[3]
        halo1_M200 = 10.**params[2]

        filament_u1_mpc = self.halo1_u_mpc
        filament_u2_mpc = self.halo2_u_mpc

        self.nh1.M_200= halo1_M200
        self.nh1.concentr = self.get_concentr(halo1_M200,self.halo1_z)

        self.nh2.M_200= halo2_M200
        self.nh2.concentr = self.get_concentr(halo2_M200,self.halo2_z)

        h1g1 , h1g2  = self.nh1.get_shears_with_pz_fast(self.shear_u_arcmin , self.shear_v_arcmin , self.grid_z_centers , self.prob_z, redshift_offset)
        h2g1 , h2g2  = self.nh2.get_shears_with_pz_fast(self.shear_u_arcmin , self.shear_v_arcmin , self.grid_z_centers , self.prob_z, redshift_offset)
        fg1 , fg2 = self.filam.filament_model_with_pz(shear_u_mpc=self.shear_u_mpc, shear_v_mpc=self.shear_v_mpc , u1_mpc=filament_u1_mpc , u2_mpc=filament_u2_mpc ,  kappa0=filament_kappa0 ,  radius_mpc=filament_radius ,  pair_z=pair_z ,  grid_z_centers=self.grid_z_centers , prob_z=self.prob_z)

        model_g1 = h1g1 + h2g1 + fg1
        model_g2 = h1g2 + h2g2 + fg2

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

            log.info('%7d post=% 2.8e like=% 2.8e prior=% 2.4e kappa0=% 6.3f radius=% 6.3f h1M200=% 5.2e h1M200=% 5.2e' % (self.n_model_evals,posterior,likelihood,prior,theta[0],theta[1],10.**theta[2],10.**theta[3]))

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
        prob=1e-10 # small number so that the prior doesn't matter

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
        model_g1 , model_g2, limit_mask = self.draw_model(theta_null)
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
        
        theta0 = []
        for iw in range(self.n_walkers):
            start = [None]*self.n_dim
            for ip in range(self.n_dim):
                start[ip]  = np.random.uniform( low=self.parameters[ip]['box']['min'] , high=self.parameters[ip]['box']['max'] ) 

            start = np.array(start)
            theta0.append( start )
                 
        
        self.sampler.run_mcmc(theta0, self.n_samples)

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

        n_grid = self.n_grid
        n_total = n_grid**self.n_dim

        self.n_model_evals = 0

        grid_kappa0 = np.linspace(self.parameters[0]['box']['min'],self.parameters[0]['box']['max'],n_grid)
        grid_radius = np.linspace(self.parameters[1]['box']['min'],self.parameters[1]['box']['max'],n_grid)
        grid_h1M200 = np.linspace(self.parameters[2]['box']['min'],self.parameters[2]['box']['max'],n_grid)
        grid_h2M200 = np.linspace(self.parameters[3]['box']['min'],self.parameters[3]['box']['max'],n_grid)
        log_post = np.zeros(n_total)
        params = np.zeros([n_total,self.n_dim])

        ia = 0
        n_models = len(grid_kappa0) * len(grid_radius)
        for ik,vk in enumerate(grid_kappa0):
            for ir,vr in enumerate(grid_radius):
                for im1,vm1 in enumerate(grid_h1M200):
                    for im2,vm2 in enumerate(grid_h2M200):
    
                            log_post[ia] = self.log_posterior([vk,vr,vm1,vm2])
                            params[ia,:] = vk , vr , vm1 , vm2
                            ia+=1
                            # if ia % 1000 == 0 : log.info('gridsearch progress %d/%d models' , ia, n_models)


        grids =  [grid_kappa0, grid_radius , grid_h1M200 , grid_h2M200]
        return log_post , params , grids 


    def get_bcc_pz(self,filename_lenscat):

        if self.prob_z == None:


            # filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
            # filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
            lenscat = tabletools.loadTable(filename_lenscat)

            if 'z' in lenscat.dtype.names:
                self.prob_z , _  = pl.histogram(lenscat['z'],bins=self.grid_z_edges,normed=True)
            elif 'z-phot' in lenscat.dtype.names:
                self.prob_z , _  = pl.histogram(lenscat['z-phot'],bins=self.grid_z_edges,normed=True)

            if 'e1' in lenscat.dtype.names:
                self.sigma_ell = np.std(lenscat['e1'],ddof=1)

    def set_shear_sigma(self):

        self.inv_sq_sigma_g = ( np.sqrt(self.shear_n_gals) / self.sigma_ell )**2


    def get_grid_max(self,log_post,params):

        imax = log_post.argmax()
        vmax_post = log_post[imax]
        # imax_kappa0 , imax_radius = np.unravel_index(log_post.argmax(), log_post.shape)
        vmax_kappa0 = params[imax,0]
        vmax_radius = params[imax,1]
        vmax_h1M200 = params[imax,2]
        vmax_h2M200 = params[imax,3]
        vmax_params = params[imax,:]
        
        log.info('ML solution log_like=% 5.2e kappa0=% 5.2f radius=% 5.2f h1M200=% 5.2e h2M200=% 5.2e', vmax_post , vmax_kappa0 , vmax_radius , 10.**vmax_h1M200 , 10.**vmax_h2M200 )

        best_model_g1, best_model_g2, limit_mask = self.draw_model( vmax_params )

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

        best_model_g1, best_model_g2, limit_mask = self.draw_model( vmax_params )

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

    shear_model_g1, shear_model_g2, limit_mask = fitobj.draw_model([0., 2., 14.5, 14.5])
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
    log.info('running mcmc search')
    fitobj.n_walkers=10
    fitobj.n_samples=2000
    fitobj.run_mcmc()
    params = fitobj.sampler.flatchain

    plotstools.plot_dist(params)
    pl.show()

    # vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_params = fitobj.get_grid_max(log_post , params)



if __name__=='__main__':

    self_fit()