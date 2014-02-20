import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools
import numpy as np
import pylab as pl


log = logging.getLogger("fil_mod_1h") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

class modelfit():

    def __init__(self):
        
        self.shear_g1 = None
        self.shear_g2 = None
        self.shear_u_arcmin = None
        self.shear_v_arcmin = None
        self.shear_z = 1
        self.halo_u_arcmin = None
        self.halo_v_arcmin = None
        self.halo_z = None
        self.sigma_g = 0.2
        self.n_model_evals = 0
        self.sampler = None
        self.gaussian_prior_theta = [{'mean' : 14, 'std': 0.5}]
        self.box_prior_theta = [{'min' : 5, 'max': 25}]


    def plot_shears(self,g1,g2,limit_mask=None):
        
        ephi=0.5*np.arctan2(g2,g1)              

        if limit_mask==None:
            emag=np.sqrt(g1**2+g2**2)
        else:
            emag=np.sqrt(g1**2+g2**2) * limit_mask


        nuse=1
        quiver_scale=20

        pl.quiver(self.shear_u_arcmin[::nuse],self.shear_v_arcmin[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale)
        
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

    def plot_residual(self,model_g1,model_g2,limit_mask=None):

        res1  = (self.shear_g1 - model_g1) 
        res2  = (self.shear_g2 - model_g2) 
        
        # emag_data=np.sqrt(self.shear_g1**2+self.shear_g2**2)  * limit_mask
        # ephi_data=0.5*np.arctan2(self.shear_g2,self.shear_g1)   * limit_mask            
        
        # emag_res=np.sqrt(res1**2+res2**2) * limit_mask
        # ephi_res=0.5*np.arctan2(res2,res1)           * limit_mask    
        
        # emag_model=np.sqrt(model_g1**2+model_g2**2)  * limit_mask
        # ephi_model=0.5*np.arctan2(model_g2,model_g1)    * limit_mask         

        # for i in range(len(res1)):

        #     log.debug( 'data=[% 2.4f % 2.4f % 2.4f % 2.4f] model=[% 2.4f % 2.4f % 2.4f % 2.4f] res=[% 2.4f % 2.4f % 2.4f % 2.4f] ' % (
        #         self.shear_g1[i], self.shear_g2[i], emag_data[i], ephi_data[i],
        #         model_g1[i], model_g2[i], emag_model[i], ephi_model[i],
        #         res1[i], res2[i], emag_res[i], ephi_res[i] ) )



        pl.figure()
        pl.subplot(3,1,1)
        self.plot_shears(self.shear_g1,self.shear_g2,limit_mask)
        pl.subplot(3,1,2)
        self.plot_shears(model_g1,model_g2,limit_mask)
        pl.subplot(3,1,3)
        self.plot_shears(res1 , res2,limit_mask)
        
    def plot_model(self,p):

        model_g1 , model_g2, limit_mask = self.draw_model(p)
        self.plot_residual(model_g1 , model_g2, limit_mask)        


    def draw_model(self,params):
        """
        params[0] mass log(M)
        """

        self.n_model_evals +=1

        M200 = 10.**params[0]
        # conc = 2.1
        conc = self.get_concentr(M200,self.halo_z)


        halo_pos = galsim.PositionD(x=self.halo_u_arcmin*60.,y=self.halo_v_arcmin*60.)
        shear_pos = ( self.shear_u_arcmin*60. , self.shear_v_arcmin*60.) 

        nfw=galsim.NFWHalo(conc=conc, redshift=self.halo_z, mass=M200, omega_m = cosmology.cospars.omega_m, halo_pos=halo_pos)
        # log.debug('getting lensing for single NFW')
        h1g1 , h1g2  =nfw.getShear(pos=shear_pos, z_s=self.shear_z)

        model_g1 = h1g1 
        model_g2 = h1g2 

        weak_limit = 0.4

        limit_mask = np.abs(model_g1 + 1j*model_g2) < weak_limit

        # self.plot_residual(model_g1,model_g2)

        pl.figure()
        pl.subplot(2,1,1)
        self.plot_shears(model_g1 , model_g2, limit_mask)
        pl.subplot(2,1,2)
        self.plot_shears_mag(model_g1 , model_g2)
        pl.suptitle('model M200=%2.4f conc=%2.4f' % (M200,conc) )
        filename_fig = 'model.%04d.png' % self.n_model_evals
        pl.savefig(filename_fig)

        return  model_g1 , model_g2 , limit_mask

    def log_posterior(self,theta):

        model_g1 , model_g2, limit_mask = self.draw_model(theta)


        likelihood = self.log_likelihood(model_g1,model_g2,limit_mask)
        prior = self.log_prior(theta)
        if not np.isfinite(prior):
            posterior = -np.inf
        else:
            posterior = likelihood + prior

        if self.n_model_evals % 100 == 1:
            log.info('%7d prob=[% 2.4e % 2.4e % 2.4e] % 2.10e' % (self.n_model_evals,posterior,likelihood,prior,theta[0]))

        if np.isnan(posterior):
            import pdb; pdb.set_trace()

        return posterior

    def log_prior(self,theta):
        
        M200 = theta[0]
        
        prob = -0.5 * ( (M200 - self.gaussian_prior_theta[0]['mean'])/self.gaussian_prior_theta[0]['std'] )**2  - np.log(np.sqrt(2*np.pi))

        if ( self.box_prior_theta[0]['min'] < M200 < self.box_prior_theta[0]['max'] ):

                return prob
        else: 
            return -np.inf
       
    def log_likelihood(self,model_g1,model_g2,limit_mask):

        res1 = (model_g1 - self.shear_g1) * limit_mask
        res2 = (model_g2 - self.shear_g2) * limit_mask
        n_points = len(np.nonzero(limit_mask))

        chi2 = -0.5 * ( np.sum( ((res1)/self.sigma_g) **2) + np.sum( ((res2)/self.sigma_g) **2) ) / n_points

        return chi2

    def run_mcmc(self):

        n_dim = 1 
        n_walkers = 10
        self.n_model_evals = 0

        log.info('getting self.sampler')
        self.sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=n_dim , lnpostfn=self.log_posterior)
        # theta0 = [ [self.gaussian_prior_theta[0]['mean'] + np.random.randn()*self.gaussian_prior_theta[0]['std']]  for i in range(n_walkers)]
        theta0 = [ [self.gaussian_prior_theta[0]['mean'] + np.random.randn()*self.gaussian_prior_theta[0]['std'] ]  for i in range(n_walkers)]
        print theta0

        self.sampler.run_mcmc(theta0, 1000)

    def run_gridsearch(self,M200_min=14,M200_max=16,M200_n=1000):

        grid_M200_min = M200_min
        grid_M200_max = M200_max
        grid_M200_n = M200_n
        log_post = np.zeros(grid_M200_n)
        grid_M200 = np.linspace(grid_M200_min,grid_M200_max,grid_M200_n)
        for im200,vm200 in enumerate(grid_M200):

            log_post[im200] = self.log_posterior([vm200])

        return log_post , grid_M200


    def get_concentr(self,M,z):

        # Duffy et al 2008 from King and Mead 2011
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cosmology.cospars.h)**(-0.081)

        print concentr
        return concentr
