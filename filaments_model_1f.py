import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools, nfw, plotstools
import numpy as np
import pylab as pl
import warnings

warnings.simplefilter('once')


log = logging.getLogger("fil_mod_1h") 
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
        
        self.shear_g1 = None
        self.shear_g2 = None
        self.shear_u_arcmin = None
        self.shear_v_arcmin = None
        self.grid_z_edges = np.linspace(0,2,10);
        self.grid_z_centers = plotstools.get_bins_centers(self.grid_z_edges)
        self.prob_z = None
        self.halo_u_arcmin = None
        self.halo_v_arcmin = None
        self.halo_z = None
        self.sigma_g = 0.2
        self.n_model_evals = 0
        self.sampler = None
        self.n_samples = 1000
        self.n_walkers = 10
        self.save_all_models = False

        self.n_dim = 2 
        self.parameters = [None]*self.n_dim
        self.parameters[0] = {}
        self.parameters[1] = {}
        self.parameters[0]['name'] = 'filament_kappa'
        self.parameters[1]['name'] = 'filament_radius'
        # self.parameters[0]['prior'] = {'mean' : 0.05, 'std': 0.01}
        # self.parameters[1]['prior'] = {'mean' : 0.1, 'std': 0.1}
        self.parameters[0]['box'] = {'min' : 0, 'max': 0.1}
        self.parameters[1]['box'] = {'min' : 0, 'max': 2}

        self.halo1_M200 = None
        self.halo1_conc = None
        self.halo1_z = None
        self.halo1_u_arcmin = None
        self.halo1_v_arcmin = None
        self.halo2_M200 = None
        self.halo2_conc = None
        self.halo2_z = None
        self.halo2_u_arcmin = None
        self.halo2_v_arcmin = None

    def plot_shears(self,g1,g2,limit_mask=None,unit='arcmin'):
        
        ephi=0.5*np.arctan2(g2,g1)              

        if limit_mask==None:
            emag=np.sqrt(g1**2+g2**2)
        else:
            emag=np.sqrt(g1**2+g2**2) * limit_mask


        nuse=1
        quiver_scale=1
        line_width=0.0005* quiver_scale

        if unit=='arcmin':
            pl.quiver(self.shear_u_arcmin[::nuse],self.shear_v_arcmin[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale , width = line_width)  
            pl.xlim([min(self.shear_u_arcmin),max(self.shear_u_arcmin)])
            pl.ylim([min(self.shear_v_arcmin),max(self.shear_v_arcmin)])
        elif unit=='Mpc':
            # not finished yet
            pl.quiver(self.shear_u_[::nuse],self.shear_v_arcmin[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale , width = line_width)  
            pl.xlim([min(self.shear_u_arcmin),max(self.shear_u_arcmin)])
            pl.ylim([min(self.shear_v_arcmin),max(self.shear_v_arcmin)])

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
        


    def draw_model(self,params):
        """
        params[0] filament_kappa
        params[1] filament_radius
        """

        self.n_model_evals +=1

        filament_kappa0 = params[0]
        filament_radius = params[1]

        filament_u1_arcmin = self.halo1_u_arcmin 
        filament_u2_arcmin = self.halo2_u_arcmin 

        fg1 , fg2 = self.filament_model(self.shear_u_arcmin,
                                        self.shear_v_arcmin,
                                        filament_u1_arcmin,
                                        filament_u2_arcmin,
                                        filament_kappa0,
                                        filament_radius)

        model_g1 = self.halos_g1 + fg1
        model_g2 = self.halos_g2 + fg2

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
            n_progress = 100
        elif log.level == logging.INFO:
            n_progress = 10000
        if self.n_model_evals % n_progress == 0:

            log.info('%7d post=% 2.4e like=% 2.4e prior=% 2.4e kappa0=% 5.3f radius=% 5.3f' % (self.n_model_evals,posterior,likelihood,prior,theta[0],theta[1]))

        if np.isnan(posterior):
            import pdb; pdb.set_trace()

        if self.save_all_models:

            self.plot_residual(model_g1,model_g2,limit_mask)

            pl.suptitle('model post=% 10.4e kappa0=%5.2e radius=%2.4f' % (posterior,theta[0],theta[1]) )
            filename_fig = 'models/res1.%04d.png' % self.n_model_evals
            pl.savefig(filename_fig,dpi=500)
            log.debug('saved %s' % filename_fig)
            pl.close()

            self.plot_residual_g1g2(model_g1,model_g2,limit_mask)

            pl.suptitle('model post=% 10.4e kappa0=%5.2e radius=%2.4f' % (posterior,theta[0],theta[1]) )
            filename_fig = 'models/res2.%04d.png' % self.n_model_evals
            pl.savefig(filename_fig,dpi=500)
            log.debug('saved %s' % filename_fig)
            pl.close()


        return posterior

    def log_prior(self,theta):
        
        filament_kappa0 = theta[0]
        filament_radius = theta[1]

        # prob = -0.5 * ( (log10_M200 - self.gaussian_prior_theta[0]['mean'])/self.gaussian_prior_theta[0]['std'] )**2  - np.log(np.sqrt(2*np.pi))
        prob=1e-10 # small number so that the prior doesn't matter

        if ( self.parameters[0]['box']['min'] <= filament_kappa0 <= self.parameters[0]['box']['max'] ):
            if ( self.parameters[1]['box']['min'] <= filament_radius <= self.parameters[1]['box']['max'] ):
                return prob

        return -np.inf
       
    def log_likelihood(self,model_g1,model_g2,limit_mask):

        res1 = (model_g1 - self.shear_g1) #* limit_mask
        res2 = (model_g2 - self.shear_g2) #* limit_mask
        # n_points = len(np.nonzero(limit_mask))

        # chi2 = -0.5 * ( np.sum( ((res1)/self.sigma_g) **2) + np.sum( ((res2)/self.sigma_g) **2) ) / n_points
        chi2 = -0.5 * ( np.sum( ((res1)/self.sigma_g) **2 ) ) 

        return chi2

    def null_log_likelihood(self):

        theta_null = [0,1]
        model_g1 , model_g2, limit_mask = self.draw_model(theta_null)
        null_log_like = self.log_likelihood(model_g1,model_g2,limit_mask)
        # null_log_post =  self.log_posterior(theta_null)
        
        # print null_log_like , null_log_post
        # pl.show()

        return null_log_like




    def run_mcmc(self):

        self.get_halo_signal()
        self.n_model_evals = 0

        log.info('getting self.sampler')


        self.sampler = emcee.EnsembleSampler(nwalkers=self.n_walkers, dim=self.n_dim , lnpostfn=self.log_posterior)
        
        theta0 = []
        for iw in range(self.n_walkers):
            start = [None]*self.n_dim
            for ip in range(self.n_dim):
                start[ip]  = np.random.uniform( low=self.parameters[ip]['box']['min'] , high=self.parameters[ip]['box']['max'] ) 

            start = np.array(start)
            theta0.append( start )
                 
        
        self.sampler.run_mcmc(theta0, self.n_samples)

    def run_gridsearch(self,n_grid=100):

        self.get_halo_signal()

        grid_kappa0_min = self.parameters[0]['box']['min']
        grid_kappa0_max = self.parameters[0]['box']['max']
        grid_radius_min = self.parameters[1]['box']['min']
        grid_radius_max = self.parameters[1]['box']['max']

        
        # log_post = np.zeros([n_grid,n_grid])

        n_total = n_grid**self.n_dim

        grid_kappa0 = np.linspace(grid_kappa0_min,grid_kappa0_max,n_grid)
        grid_radius = np.linspace(grid_radius_min,grid_radius_max,n_grid)
        log_post = np.zeros(n_total)
        params = np.zeros([n_total,self.n_dim])

        ia = 0
        n_models = len(grid_kappa0) * len(grid_radius)
        for ik,vk in enumerate(grid_kappa0):
            for ir,vr in enumerate(grid_radius):
    
                # log_post[ir,ik] = self.log_posterior([vk,vr])
                log_post[ia] = self.log_posterior([vk,vr])
                params[ia,:] = vk , vr


                # if ia % 1000 == 0 : log.info('gridsearch progress %d/%d models' , ia, n_models)
                ia+=1


        return log_post , params , grid_kappa0, grid_radius

    def get_grid_max(self,log_post,params):

        imax = log_post.argmax()
        vmax_post = log_post[imax]
        # imax_kappa0 , imax_radius = np.unravel_index(log_post.argmax(), log_post.shape)
        vmax_kappa0 = params[imax,0]
        vmax_radius = params[imax,1]
        
        log.info('maximum likelihood solution kappa0=% 5.2e radius=% 5.2e log_like=% 5.2e',vmax_kappa0,vmax_radius, vmax_post )

        best_model_g1, best_model_g2, limit_mask = self.draw_model( [vmax_kappa0 , vmax_radius] )

        return  vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_kappa0 , vmax_radius





    def get_concentr(self,M,z):

        # Duffy et al 2008 from King and Mead 2011
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cosmology.cospars.h)**(-0.081)

        return concentr

    def get_bcc_pz(self):

        if self.prob_z == None:


            filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
            lenscat = tabletools.loadTable(filename_lenscat)
            self.prob_z , _  = pl.histogram(lenscat['z'],bins=self.grid_z_edges,normed=True)
  

    def filament_model(self,shear_u_arcmin,shear_v_arcmin,u1_arcmin,u2_arcmin,kappa0,radius_arcmin):

        r = np.abs(shear_v_arcmin)

        kappa = - kappa0 / (1. + r / radius_arcmin)

        # zero the filament outside halos
        # we shoud zero it at R200, but I have to check how to calculate it from M200 and concentr
        select = ((shear_u_arcmin > (u1_arcmin) ) + (shear_u_arcmin < (u2_arcmin)  ))
        kappa[select] *= 0.
        g1 = kappa
        g2 = g1*0.

        return g1 , g2




