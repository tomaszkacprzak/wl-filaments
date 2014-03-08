import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools, nfw, plotstools
import numpy as np
import pylab as pl


log = logging.getLogger("fil_mod_1h") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

redshift_offset = 0.2

class modelfit():

    def __init__(self):
        
        self.shear_g1 = None
        self.shear_g2 = None
        self.shear_u_arcmin = None
        self.shear_v_arcmin = None
        self.grid_z_edges = np.linspace(0,2,10);
        self.grid_z_centers = plotstools.get_bins_centers(self.grid_z_edges)
        self.prob_z = np.ones_like(self.grid_z_centers) / len(self.grid_z_centers)
        self.halo_u_arcmin = None
        self.halo_v_arcmin = None
        self.halo_z = None
        self.sigma_g = 0.2
        self.n_model_evals = 0
        self.sampler = None
        self.n_samples = 1000
        self.n_walkers = 10
        self.save_all_models = False

        self.n_dim = 4 
        self.parameters = [None]*self.n_dim
        self.parameters[0] = {}
        self.parameters[1] = {}
        self.parameters[2] = {}
        self.parameters[3] = {}
        self.parameters[0]['name'] = 'halo1_M200'
        self.parameters[1]['name'] = 'halo2_M200'
        self.parameters[2]['name'] = 'filament_kappa'
        self.parameters[3]['name'] = 'filament_radius'
        self.parameters[0]['prior'] = {'mean' : 14, 'std': 1}
        self.parameters[1]['prior'] = {'mean' : 14, 'std': 1}
        self.parameters[2]['prior'] = {'mean' : 0.05, 'std': 0.01}
        self.parameters[3]['prior'] = {'mean' : 0.1, 'std': 0.1}
        self.parameters[3]['box'] = {'min' : 5, 'max': 25}
        self.parameters[3]['box'] = {'min' : 5, 'max': 25}
        self.parameters[3]['box'] = {'min' : 0, 'max': 0.1}
        self.parameters[3]['box'] = {'min' : 0, 'max': 2}

    def plot_shears(self,g1,g2,limit_mask=None):
        
        ephi=0.5*np.arctan2(g2,g1)              

        if limit_mask==None:
            emag=np.sqrt(g1**2+g2**2)
        else:
            emag=np.sqrt(g1**2+g2**2) * limit_mask


        nuse=1
        quiver_scale=3

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

        M200 = 10**params[0]
        # conc = 2.1
        conc = self.get_concentr(M200,self.halo_z)

        # halo_pos = galsim.PositionD(x=self.halo_u_arcmin*60.,y=self.halo_v_arcmin*60.)
        # shear_pos = ( self.shear_u_arcmin*60. , self.shear_v_arcmin*60.) 

        # nfw=galsim.NFWHalo(conc=conc, redshift=self.halo_z, mass=M200, omega_m = cosmology.cospars.Omega_m, halo_pos=halo_pos)

        nh = nfw.NfwHalo()
        nh.M_200=M200
        nh.concentr=conc
        nh.z_cluster= self.halo_z
        nh.theta_cx = self.halo_u_arcmin
        nh.theta_cy = self.halo_v_arcmin 

        list_h1g1 = []
        list_h1g2 = []
        list_weight = []

        for ib, vb in enumerate(self.grid_z_centers):
            if vb < (self.halo_z+redshift_offset): continue
            [h1g1 , h1g2 , Delta_Sigma_1, Delta_Sigma_2 , Sigma_crit]=nh.get_shears(self.shear_u_arcmin , self.shear_v_arcmin , vb)
            weight = self.prob_z[ib]
            list_h1g1.append(h1g1*weight) 
            list_h1g2.append(h1g2*weight) 
            list_weight.append(weight)

        h1g1 = np.sum(np.array(list_h1g1),axis=0) / np.sum(list_weight)
        h1g2 = np.sum(np.array(list_h1g2),axis=0) / np.sum(list_weight)

        # median_redshift = 0.8
        # [h1g1 , h1g2 , Delta_Sigma_1, Delta_Sigma_2 , Sigma_crit]=nh.get_shears(self.shear_u_arcmin , self.shear_v_arcmin , median_redshift )

        # print Sigma_crit
        # model_g1 = h1g1 * Sigma_crit
        # model_g2 = h1g2 * Sigma_crit

        model_g1 = h1g1
        model_g2 = h1g2

        weak_limit = 0.2

        limit_mask = np.abs(model_g1 + 1j*model_g2) < weak_limit
        n_outside_weak_limit=len(limit_mask)-sum(limit_mask)

        log.debug('draw_model: max=%2.2f n_outside_weak_limit %d' % (model_g1.max(),n_outside_weak_limit))  


        # self.plot_residual(model_g1,model_g2)

        if self.save_all_models:
            pl.figure()
            pl.subplot(2,1,1)
            self.plot_shears(model_g1 , model_g2, limit_mask)
            pl.subplot(2,1,2)
            self.plot_shears_mag(model_g1 , model_g2)
            pl.suptitle('model M200=%5.2e conc=%2.4f' % (M200,conc) )
            filename_fig = 'model.%04d.png' % self.n_model_evals
            pl.savefig(filename_fig)
            pl.close()
            log.debug('saved %s' % filename_fig)

        return  model_g1 , model_g2 , limit_mask

    def log_posterior(self,theta):

        model_g1 , model_g2, limit_mask = self.draw_model(theta)


        likelihood = self.log_likelihood(model_g1,model_g2,limit_mask)
        prior = self.log_prior(theta)
        if not np.isfinite(prior):
            posterior = -np.inf
        else:
            posterior = likelihood 

        if self.n_model_evals % 100 == 1:
            log.info('%7d prob=[% 2.4e % 2.4e % 2.4e] % 2.10e' % (self.n_model_evals,posterior,likelihood,prior,theta[0]))

        if np.isnan(posterior):
            import pdb; pdb.set_trace()

        return posterior

    def log_prior(self,theta):
        
        log10_M200 = theta[0]
        
        prob = -0.5 * ( (log10_M200 - self.gaussian_prior_theta[0]['mean'])/self.gaussian_prior_theta[0]['std'] )**2  - np.log(np.sqrt(2*np.pi))

        if ( self.box_prior_theta[0]['min'] < log10_M200 < self.box_prior_theta[0]['max'] ):

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

        self.n_model_evals = 0

        log.info('getting self.sampler')
        self.sampler = emcee.EnsembleSampler(nwalkers=self.n_walkers, dim=self.n_dim , lnpostfn=self.log_posterior)
        # theta0 = [ [self.gaussian_prior_theta[0]['mean'] + np.random.randn()*self.gaussian_prior_theta[0]['std']]  for i in range(n_walkers)]
        theta0 = [ [self.gaussian_prior_theta[0]['mean'] + np.random.randn()*self.gaussian_prior_theta[0]['std'] ]  for i in range(n_walkers)]
        print theta0

        self.sampler.run_mcmc(theta0, self.n_samples)

    def run_gridsearch(self,M200_min=14,M200_max=15.3,M200_n=100):

        grid_M200_min = M200_min
        grid_M200_max = M200_max
        grid_M200_n = M200_n
        log_post = np.zeros(grid_M200_n)
        grid_M200 = np.linspace(grid_M200_min,grid_M200_max,grid_M200_n)
        for im200,vm200 in enumerate(grid_M200):

            # log.info('%5d m200=%2.2e' %(im200,M200))
            log_post[im200] = self.log_posterior([vm200])

        imax = log_post.argmax()
        vmax = grid_M200[imax]
        M200 = 10**vmax
        conc = self.get_concentr(M200,self.halo_z)
        log.info('maximum likelihood solution M200=% 5.2e conc=% 5.2f log_like=% 5.2e %d',M200,conc,log_post[imax],imax)

        best_model_g1, best_model_g2, limit_mask = self.draw_model([grid_M200[imax]])

        return log_post , grid_M200, best_model_g1, best_model_g2 , limit_mask


    def get_concentr(self,M,z):

        # Duffy et al 2008 from King and Mead 2011
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cosmology.cospars.h)**(-0.081)

        return concentr

    def get_bcc_pz(self):

        filename_lenscat = '/home/tomek/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
        lenscat = tabletools.loadTable(filename_lenscat)

        self.prob_z , _ , _ = pl.hist(lenscat['z'],bins=self.grid_z_edges,normed=True)


    
























class fitter():

    shear_g1 = None
    shear_g2 = None
    shear_u_arcmin = None
    shear_v_arcmin = None
    shear_z = 1000
    pairs_u1_arcmin = None
    pairs_v1_arcmin = None
    pairs_u2_arcmin = None
    pairs_v2_arcmin = None
    pairs_z1 = None
    pairs_z2 = None
    sigma_g = 0.2


    def draw_model(self,params):
        """
        params[0] mass log(M1)
        params[1] mass log(M2)
        params[2] filament_radius 
        params[3] filament_kappa 
        """

        # def draw_model(M1,M2,F,R,shears_info,pair_info):

        M1 = np.exp(params[0])
        M2 = np.exp(params[1])
        filament_radius = params[2]
        filament_kappa = params[3]

        c1 = self.get_concentr(M1,self.pairs_z1)
        c2 = self.get_concentr(M2,self.pairs_z2)
        halo1_pos = galsim.PositionD(x=self.pairs_u1_arcmin*60.,y=self.pairs_v1_arcmin*60.)
        halo2_pos = galsim.PositionD(x=self.pairs_u2_arcmin*60.,y=self.pairs_v2_arcmin*60.)
        shear_pos = ( self.shear_u_arcmin*60. , self.shear_v_arcmin*60.) 

        nfw1=galsim.NFWHalo(conc=c1, redshift=self.pairs_z1, mass=M1, omega_m = cosmology.cospars.omega_m, halo_pos=halo1_pos)
        nfw2=galsim.NFWHalo(conc=c2, redshift=self.pairs_z2, mass=M2, omega_m = cosmology.cospars.omega_m, halo_pos=halo2_pos)
        log.debug('getting lensing')
        h1g1 , h1g2 , _ =nfw1.getLensing(pos=shear_pos, z_s=self.shear_z)
        h2g1 , h2g2 , _ =nfw2.getLensing(pos=shear_pos, z_s=self.shear_z)
        fg1 , fg2 = self.filament_model(self.shear_u_arcmin,self.shear_v_arcmin,self.pairs_u1_arcmin,self.pairs_u2_arcmin,filament_kappa,filament_radius)

        model_g1 = h1g1 + h2g1 + fg1
        model_g2 = h1g2 + h2g2 + fg2
        return  model_g1 , model_g2 

    def lnpostfn(self,p):

        model_g1 , model_g2 = self.draw_model(p)
        chi2 = self.log_posterior(model_g1,model_g2)

    def log_posterior(self,model_g1,model_g2):

        chi2 = - np.sum( ((model_g1 - self.shear_g1)/self.sigma_g) **2) + np.sum( ((model_g2 - self.shear_g2)/self.sigma_g) **2)

        print chi2
        return chi2

    def run_mcmc(self):

        n_dim = 4
        n_walkers = 10

        log.info('getting sampler')
        sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=n_dim , lnpostfn=self.lnpostfn, a=2.0, args=[], threads=1, pool=None, live_dangerously=False)
        p0_walker= np.array([ 13.0, 14.0, 0.001, 0.05 ])
        p0 = [p0_walker for i in range(n_walkers)]
        sampler.run_mcmc(p0, 1000)

    def get_concentr(self,M,z):

        # Duffy et al 2008 from King and Mead 2011
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cospars.h)**(-0.081)
        return concentr

    def filament_model(self,shear_u_arcmin,shear_v_arcmin,u1_arcmin,u2_arcmin,kappa0,radius_arcmin):

        r = np.abs(shear_v_arcmin)

        kappa = - kappa0 / (1. + r / radius_arcmin)

        # zero the filament outside halos
        # we shoud zero it at R200, but I have to check how to calculate it from M200 and concentr
        select = ((shear_u_arcmin > u1_arcmin) + (shear_u_arcmin < u2_arcmin))
        kappa[select] *= 0.
        g1 = kappa
        g2 = g1*0.

        return g1 , g2




