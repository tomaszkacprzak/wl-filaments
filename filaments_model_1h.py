import yaml, argparse, sys, logging , pyfits, emcee, tabletools, cosmology, filaments_tools, nfw, plotstools, warnings
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
        self.n_dim = 1 
        self.n_walkers = 10
        self.gaussian_prior_theta = [{'mean' : 14, 'std': 1}]
        self.box_prior_theta = [{'min' : 5, 'max': 25}]
        self.save_all_models = False
        self.mean_inv_sigma_crit = None
        self.grid_z_edges = np.linspace(0,2,10);
        self.grid_z_centers = plotstools.get_bins_centers(self.grid_z_edges)

    def get_bcc_pz(self):

        if self.prob_z == None:


            filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
            lenscat = tabletools.loadTable(filename_lenscat)
            self.prob_z , _  = pl.histogram(lenscat['z'],bins=self.grid_z_edges,normed=True)




    def plot_shears(self,g1,g2,limit_mask=None):
        
        ephi=0.5*np.arctan2(g2,g1)              

        if limit_mask==None:
            emag=np.sqrt(g1**2+g2**2)
        else:
            emag=np.sqrt(g1**2+g2**2) * limit_mask


        nuse=1
        quiver_scale=1
        pl.quiver(self.shear_u_arcmin[::nuse],self.shear_v_arcmin[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale)
        
        pl.xlim([min(self.shear_u_arcmin),max(self.shear_u_arcmin)])
        pl.ylim([min(self.shear_v_arcmin),max(self.shear_v_arcmin)])
        pl.axis('image')

    def plot_shears_g(self,g):

        # emag=np.sqrt(g1**2+g2**2)

        
        pl.scatter(self.shear_u_arcmin,self.shear_v_arcmin,30,c=g,marker='s',edgecolor=None, lw = 0)
        pl.xlim([min(self.shear_u_arcmin),max(self.shear_u_arcmin)])
        pl.ylim([min(self.shear_v_arcmin),max(self.shear_v_arcmin)])
        pl.axis('image')
        pl.colorbar()

    def plot_shears_all(self,model_g1,model_g2,limit_mask=None):

        pl.figure(figsize=(30,10))
        pl.subplot(1,3,1)
        self.plot_shears(model_g1 , model_g2, limit_mask)
        pl.subplot(1,3,2)
        self.plot_shears_g(model_g1)
        pl.subplot(1,3,3)
        self.plot_shears_g(model_g2)


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
        conc = self.get_concentr(M200,self.halo_z)

        # halo_pos = galsim.PositionD(x=self.halo_u_arcmin*60.,y=self.halo_v_arcmin*60.)
        # shear_pos = ( self.shear_u_arcmin*60. , self.shear_v_arcmin*60.) 

        # nfw=galsim.NFWHalo(conc=conc, redshift=self.halo_z, mass=M200, omega_m = cosmology.cospars.Omega_m, halo_pos=halo_pos)

        # nh = nfw.NfwHalo()
        self.nh.M_200=M200
        self.nh.concentr=conc
        # nh.z_cluster= self.halo_z
        # nh.theta_cx = self.halo_u_arcmin
        # nh.theta_cy = self.halo_v_arcmin 

        list_h1g1 = []
        list_h1g2 = []
        list_weight = []

        h1g1 , h1g2  = self.nh.get_shears_with_pz_fast(self.shear_u_arcmin , self.shear_v_arcmin , self.grid_z_centers , self.prob_z, redshift_offset)

        model_g1 = h1g1
        model_g2 = h1g2

        weak_limit = 0.2

        limit_mask = np.abs(model_g1 + 1j*model_g2) < weak_limit
        n_outside_weak_limit=len(limit_mask)-sum(limit_mask)

        # log.debug('draw_model: max=%2.2f n_outside_weak_limit %d' % (model_g1.max(),n_outside_weak_limit))  


        # self.plot_residual(model_g1,model_g2)

        if self.save_all_models:
            self.plot_shears_all(model_g1,model_g2,limit_mask)

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

        if self.n_model_evals % 10 == 0:
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

        res1_sq = ((model_g1 - self.shear_g1)**2) * self.inv_sq_sigma_g #* limit_mask
        res2_sq = ((model_g2 - self.shear_g2)**2) * self.inv_sq_sigma_g #* limit_mask
        # n_points = len(np.nonzero(limit_mask))

        # chi2 = -0.5 * ( np.sum( ((res1)/self.sigma_g) **2) + np.sum( ((res2)/self.sigma_g) **2) ) / n_points
        chi2 = -0.5 * ( np.sum( res1_sq )  + np.sum( res2_sq  ) )

        return chi2


    def run_mcmc(self):

        self.n_model_evals = 0

        log.info('getting self.sampler')
        self.sampler = emcee.EnsembleSampler(nwalkers=self.n_walkers, dim=self.n_dim , lnpostfn=self.log_posterior)
        # theta0 = [ [self.gaussian_prior_theta[0]['mean'] + np.random.randn()*self.gaussian_prior_theta[0]['std']]  for i in range(n_walkers)]
        theta0 = [ [self.gaussian_prior_theta[0]['mean'] + np.random.randn()*self.gaussian_prior_theta[0]['std'] ]  for i in range(self.n_walkers)]
        print theta0

        self.sampler.run_mcmc(theta0, self.n_samples)

    def run_gridsearch(self,M200_min=14,M200_max=15.3,M200_n=100):

        grid_M200_min = M200_min
        grid_M200_max = M200_max
        grid_M200_n = M200_n
        log_post = np.zeros(grid_M200_n)
        grid_M200 = np.linspace(grid_M200_min,grid_M200_max,grid_M200_n)

        self.nh = nfw.NfwHalo()
        self.nh.z_cluster= self.halo_z
        self.nh.theta_cx = self.halo_u_arcmin
        self.nh.theta_cy = self.halo_v_arcmin 
        self.nh.set_mean_inv_sigma_crit(self.grid_z_centers,self.prob_z,self.halo_z)

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
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14 )**(-0.081)

        # warnings.warn('using pre-set concentration')
        # concentr = self.conc        

        return concentr

    def get_bcc_pz(self,filename_lenscat=None):

        if filename_lenscat==None:
            filename_lenscat = '/home/tomek/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
        lenscat = tabletools.loadTable(filename_lenscat)

        pl.figure()
        self.prob_z , _ , _ = pl.hist(lenscat['z'],bins=self.grid_z_edges,normed=True)

        if 'e1' in lenscat.dtype.names:
            select = lenscat['star_flag'] == 0
            lenscat = lenscat[select]
            select = lenscat['fitclass'] == 0
            lenscat = lenscat[select]
            select = (lenscat['e1'] != 0.0) * (lenscat['e2'] != 0.0)
            lenscat = lenscat[select]
            self.sigma_ell = np.std(lenscat['e1']*lenscat['weight'],ddof=1)


    def set_shear_sigma(self):

        self.inv_sq_sigma_g = ( np.sqrt(self.shear_w) / self.sigma_ell )**2

        # remove nans -- we shouldn't have nans in the data, but we appear to have
        select = np.isnan(self.shear_g1) | np.isnan(self.shear_g2)
        self.inv_sq_sigma_g[select] = 0
        self.shear_g1[select] = 0
        self.shear_g2[select] = 0
        n_nans = sum(np.isnan(self.shear_g1))
        log.info('found %d nan pixels' % n_nans)


def get_post_from_log(log_post):

    log_post = log_post - max(log_post.flatten())
    post = np.exp(log_post)
    norm = np.sum(post)
    post = post / norm

    return post



    
if __name__ == '__main__':

    id_pair = 4
    filename_shears = 'shears_bcc_g.fits' 
    filename_pairs = 'pairs_bcc.fits'
    filename_halo1 = 'pairs_bcc.halos1.fits'
    filename_halo2 = 'pairs_bcc.halos2.fits'

    pairs_table = tabletools.loadTable(filename_pairs)
    shears_info = tabletools.loadTable(filename_shears,hdu=id_pair+1)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    true1_M200 = np.log10(halo1_table['m200'][id_pair])
    true2_M200 = np.log10(halo2_table['m200'][id_pair])
    conc1 = halo1_table['r200'][id_pair]/halo1_table['rs'][id_pair]*1000.
    conc2 = halo2_table['r200'][id_pair]/halo2_table['rs'][id_pair]*1000.
    log.info( 'halo1 M200 %5.2e',halo1_table['m200'][id_pair] )
    log.info( 'halo2 M200 %5.2e',halo2_table['m200'][id_pair] )
    log.info( 'halo1 conc %5.2f',conc1)
    log.info( 'halo2 conc %5.2f',conc2)


    fitobj = modelfit()
    fitobj.get_bcc_pz()
    fitobj.conc = conc1
    fitobj.sigma_g =  0.001
    fitobj.shear_g1 =  shears_info['g1'] + np.random.randn(len(shears_info['g1']))*fitobj.sigma_g
    fitobj.shear_g2 =  shears_info['g2'] + np.random.randn(len(shears_info['g2']))*fitobj.sigma_g
    fitobj.shear_u_arcmin =  shears_info['u_arcmin']
    fitobj.shear_v_arcmin =  shears_info['v_arcmin']
    fitobj.halo_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
    fitobj.halo_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
    fitobj.halo_z =  pairs_table['z'][id_pair]
    fitobj.plot_shears_mag(fitobj.shear_g1,fitobj.shear_g2)
    pl.show()

    pair_info = pairs_table[id_pair]

    log.info('running grid search')
    log_post , grid_M200, best_g1, best_g2, best_limit_mask = fitobj.run_gridsearch(M200_min=13.5,M200_max=16,M200_n=1000)
    prob_post = get_post_from_log(log_post)
    pl.figure()
    pl.plot(grid_M200 , log_post , '.-')
    pl.figure()
    pl.plot(grid_M200 , prob_post , '.-')
    plotstools.adjust_limits()
    pl.figure()
    fitobj.plot_residual(best_g1, best_g2)
    pl.show()


    log.info('running mcmc - type c to continue')
    import pdb; pdb.set_trace()
    fitobj.run_mcmc()
    print fitobj.sampler
    pl.figure()
    pl.hist(fitobj.sampler.flatchain, bins=np.linspace(13,16,100), color="k", histtype="step")
    # pl.plot(fitobj.sampler.flatchain,'x')
    pl.show()
    median_m = [np.median(fitobj.sampler.flatchain)]
    print median_m
    fitobj.plot_model(median_m)
    filename_fig = 'halo_model_median.png'
    pl.savefig(filename_fig)
    log.info('saved %s' % filename_fig)


    pl.figure()
    pl.hist(fitobj.sampler.flatchain, 100, color="k", histtype="step")
    pl.show()

    median_m = [np.median(fitobj.sampler.flatchain)]
    print 'median_m %2.2e' % 10**median_m
    fitobj.plot_model(median_m)


    import pdb;pdb.set_trace()



