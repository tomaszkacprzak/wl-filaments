import os, yaml, argparse, sys, logging , pyfits,  emcee, tabletools, cosmology, filaments_tools, nfw, plotstools, filament
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
weak_limit = 1000

dtype_stats = {'names' : ['id','kappa0_signif', 'kappa0_map', 'kappa0_err_hi', 'kappa0_err_lo', 'radius_map',    'radius_err_hi', 'radius_err_lo', 'chi2_red_null', 'chi2_red_max',  'chi2_red_D', 'chi2_red_LRT' , 'chi2_null', 'chi2_max', 'chi2_D' , 'chi2_LRT' ,'sigma_g' ] , 
        'formats' : ['i8'] + ['f8']*16 }


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
        self.parameters[1]['name'] = 'filament_radius' # Mpc
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

        model_g1 , model_g2, limit_mask , _ , _  = self.draw_model(p)
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
        params[1] filament_radius [Mpc]
        """

        self.n_model_evals +=1

        filament_kappa0 = params[0]
        filament_radius = params[1]

        filament_u1_mpc = self.halo1_u_mpc
        filament_u2_mpc = self.halo2_u_mpc

        pair_z = np.mean([self.halo1_z, self.halo2_z])

        # fg1 , fg2 = self.filam.filament_model_with_pz_kron(self.shear_u_mpc, self.shear_v_mpc ,  filament_u1_mpc ,  filament_u2_mpc ,  filament_kappa0 ,  filament_radius ,  pair_z ,  self.grid_z_centers , self.prob_z)
        fg1 , fg2, DeltaSigma , SigmaCrit , kappa = self.filam.filament_model_with_pz(self.shear_u_mpc, self.shear_v_mpc ,  filament_u1_mpc ,  filament_u2_mpc ,  filament_kappa0 ,  filament_radius ,  pair_z ,  self.grid_z_centers , self.prob_z)
        # fg1 , fg2 = self.filam.fast_filament_model_with_pz(self.shear_u_mpc, self.shear_v_mpc ,  filament_u1_mpc ,  filament_u2_mpc ,  filament_kappa0 ,  filament_radius ,  pair_z ,  self.grid_z_centers , self.prob_z)


        model_g1 = self.halos_g1 + fg1
        model_g2 = self.halos_g2 + fg2

        limit_mask = np.abs(model_g1 + 1j*model_g2) < weak_limit

        return  model_g1 , model_g2 , limit_mask , DeltaSigma , kappa 

    def log_posterior(self,theta):


        model_g1 , model_g2, limit_mask , _ , _ = self.draw_model(theta)


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
        model_g1 , model_g2, limit_mask , _ , _ = self.draw_model(theta_null)
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

        self.filam = filament.filament()
        self.filam.pair_z = (self.halo1_z + self.halo2_z) / 2. 
        self.filam.grid_z_centers = self.grid_z_centers
        self.filam.prob_z = self.prob_z
        self.filam.set_mean_inv_sigma_crit(self.filam.grid_z_centers,self.filam.prob_z,self.filam.pair_z)

        # self.filam.n_points = len(self.shear_u_mpc)
        # self.filam.n_sigma_crit = len(self.grid_z_centers)
        # self.filam.get_sigma_crit_krons(n_points=len(self.shear_u_mpc))
        # self.filam.get_sigma_crit_krons(n_points=1000)

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

        best_model_g1, best_model_g2, limit_mask , _ , _ = self.draw_model( [vmax_kappa0 , vmax_radius] )

        return  vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_kappa0 , vmax_radius


    def get_bcc_pz(self):

        if self.prob_z == None:


            filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
            lenscat = tabletools.loadTable(filename_lenscat)
            self.prob_z , _  = pl.histogram(lenscat['z'],bins=self.grid_z_edges,normed=True)
  

    # def filament_model(self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z):
    #     """
    #     @brief create filament model
    #     @param shear_u_mpc array of u coordinates
    #     @param shear_v_mpc array of v coordinates
    #     @param u1_mpc start of the filament 
    #     @param u2_mpc end of the filament 
    #     @param kappa0 total mass of the filament 
    #     """

    #     # profile             :  A / (1+r**2/rc**2) )
    #     # kappa0=total mass :  r * pi * A

    #     amplitude = kappa0 /  (r * np.pi) 

    #     r = np.abs(shear_v_mpc)

    #     kappa = - amplitude / (1. + (r / radius_mpc)**2 )

    #     # zero the filament outside halos
    #     # we shoud zero it at R200, but I have to check how to calculate it from M200 and concentr
    #     select = ((shear_u_mpc > (u1_mpc) ) + (shear_u_mpc < (u2_mpc)  ))
    #     kappa[select] *= 0.
    #     g1 = kappa
    #     g2 = g1*0.


    #     return g1 , g2

    #     # 1/ (1+ r**2/rc**2) 
    #     # pi *rc * arctan(r/rc) + 1/2

    #     #  r  == tan( (0.25 - 1/2) / pi / rc ) / rc

    # def filament_model_with_pz(shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z, grid_z_centers , prob_z,  redshift_offset=0.2):

    #     list_h1g1 = []
    #     list_h1g2 = []

    #     for ib, vb in enumerate(grid_z_centers):
    #         if vb < (pair_z+redshift_offset): continue
    #         [h1g1 , h1g2 , Delta_Sigma_1, Delta_Sigma_2 , Sigma_crit]= filament_model(,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc)
    #         weight = prob_z[ib]
    #         list_h1g1.append(h1g1*weight) 
    #         list_h1g2.append(h1g2*weight) 
    #         list_weight.append(weight)

    #     h1g1 = np.sum(np.array(list_h1g1),axis=0) / np.sum(list_weight)
    #     h1g2 = np.sum(np.array(list_h1g2),axis=0) / np.sum(list_weight)

    #     return h1g1, h1g2



















def fit_single_filament(save_plots=False):

    if args.first == -1: 
        id_pair_first = 0 ; 
    else: 
        id_pair_first = args.first
    
    id_pair_last = args.first + args.num 

    log.info('running on pairs from %d to %d' , id_pair_first , id_pair_last)

    filename_pairs = config['filename_pairs']                                  
    filename_halo1 = config['filename_pairs'].replace('.fits' , '.halos1.fits')
    filename_halo2 = config['filename_pairs'].replace('.fits' , '.halos1.fits')
    filename_shears =config['filename_shears']                                 

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    filename_results_prob = 'results.prob.%04d.%04d.' % (id_pair_first, id_pair_last) + filename_shears.replace('.fits','.pp2')
    filename_results_grid = 'results.grid.%04d.%04d.' % (id_pair_first, id_pair_last) + filename_shears.replace('.fits','.pp2')
    filename_results_pairs = 'results.stats.%04d.%04d.' % (id_pair_first, id_pair_last) + filename_shears.replace('.fits','.cat')
    if os.path.isfile(filename_results_prob):
        os.remove(filename_results_prob)
        log.warning('overwriting file %s ' , filename_results_prob)
    if os.path.isfile(filename_results_pairs):
        os.remove(filename_results_pairs)
        log.warning('overwriting file %s ' , filename_results_pairs)
    if os.path.isfile(filename_results_grid):
        os.remove(filename_results_grid)
        log.warning('overwriting file %s ' , filename_results_grid)

    tabletools.writeHeader(filename_results_pairs,filaments_model_1f.dtype_stats)
    

    # get prob_z
    fitobj = filaments_model_1f.modelfit()
    fitobj.get_bcc_pz()
    prob_z = fitobj.prob_z


    # empty container list for probability measurements
    # table_stats = np.zeros(len(range(id_pair_first,id_pair_last)) , dtype=dtype_stats)

    for id_pair in range(id_pair_first,id_pair_last):

        # now we use that
        shears_info = tabletools.loadTable(filename_shears,hdu=id_pair+1)

        log.info('--------- pair %d with %d shears--------' , id_pair , len(shears_info)) 


        if len(shears_info) > 50000:
            log.warning('buggy pair, n_shears=%d , skipping' , len(shears_info))
            result_dict = {'id' : id_pair}
            tabletools.savePickle(filename_results_prob,prob_result,append=True)
            continue


        true_M200 = np.log10(halo1_table['m200'][id_pair])
        true_M200 = np.log10(halo2_table['m200'][id_pair])

        halo1_conc = halo1_table['r200'][id_pair]/halo1_table['rs'][id_pair]*1000.
        halo2_conc = halo2_table['r200'][id_pair]/halo2_table['rs'][id_pair]*1000.

        log.info( 'M200=[ %1.2e , %1.2e ] , conc=[ %1.2f , %1.2f ]',halo1_table['m200'][id_pair] , halo2_table['m200'][id_pair] , halo1_conc , halo2_conc)
        
        sigma_g_add =  config['sigma_add']

        fitobj = filaments_model_1f.modelfit()
        fitobj.prob_z = prob_z
        fitobj.shear_u_arcmin =  shears_info['u_arcmin']
        fitobj.shear_v_arcmin =  shears_info['v_arcmin']
        fitobj.shear_u_mpc =  shears_info['u_mpc']
        fitobj.shear_v_mpc =  shears_info['v_mpc']
        fitobj.shear_g1 =  shears_info['g1'] + np.random.randn(len(shears_info['g1']))*sigma_g_add
        fitobj.shear_g2 =  shears_info['g2'] + np.random.randn(len(shears_info['g2']))*sigma_g_add
        fitobj.sigma_g =  np.std(shears_info['g2'],ddof=1)

        log.info('using sigma_g=%2.5f' , fitobj.sigma_g)
        
        fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
        fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
        fitobj.halo1_u_mpc =  pairs_table['u1_mpc'][id_pair]
        fitobj.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair]
        fitobj.halo1_z =  pairs_table['z'][id_pair]
        fitobj.halo1_M200 = halo1_table['m200'][id_pair]
        fitobj.halo1_conc = halo1_conc

        fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
        fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
        fitobj.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair]
        fitobj.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair]
        fitobj.halo2_z =  pairs_table['z'][id_pair]
        fitobj.halo2_M200 = halo2_table['m200'][id_pair]
        fitobj.halo2_conc = halo2_conc

        fitobj.parameters[0]['box']['min'] = 0.
        fitobj.parameters[0]['box']['max'] = 1
        fitobj.parameters[1]['box']['min'] = 0.001
        fitobj.parameters[1]['box']['max'] = 10
        
        # fitobj.plot_shears_mag(fitobj.shear_g1,fitobj.shear_g2)
        # pl.show()
        # fitobj.save_all_models=False
        log.info('running grid search')
        n_grid=config['n_grid']
        log_post , params, grid_kappa0, grid_radius = fitobj.run_gridsearch(n_grid=n_grid)

        # get the normalised PDF and use the same normalisation on the log
        prob_post , _ , _ , _ = mathstools.get_normalisation(log_post)
        # get the maximum likelihood solution
        vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_kappa0 , vmax_radius = fitobj.get_grid_max(log_post,params)
        # get the marginals
        prob_post_matrix = np.reshape(  prob_post , [n_grid, n_grid] )    
        prob_post_kappa0 = prob_post_matrix.sum(axis=1)
        prob_post_radius = prob_post_matrix.sum(axis=0)
        # get confidence intervals on kappa0
        max_kappa0 , kappa0_err_hi , kappa0_err_lo = mathstools.estimate_confidence_interval(grid_kappa0 , prob_post_kappa0)
        max_radius , radius_err_hi , radius_err_lo = mathstools.estimate_confidence_interval(grid_radius , prob_post_radius)
        err_use = np.mean([kappa0_err_hi,kappa0_err_lo])
        kappa0_significance = max_kappa0/err_use
        # get the likelihood test
        v_reducing_null = len(fitobj.shear_u_arcmin) - 1
        v_reducing_mod = len(fitobj.shear_u_arcmin) - 2 - 1
        chi2_null = fitobj.null_log_likelihood()
        chi2_max = vmax_post
        chi2_red_null = chi2_null / v_reducing_null
        chi2_red_max = chi2_max / v_reducing_mod
        chi2_D = 2*(chi2_max - chi2_null)
        chi2_D_red = 2*(chi2_red_max - chi2_red_null)
        ndof=2
        chi2_LRT_red = 1. - scipy.stats.chi2.cdf(chi2_D_red, ndof)
        chi2_LRT = 1. - scipy.stats.chi2.cdf(chi2_D, ndof)
        # likelihood_ratio_test = scipy.stats.chi2.pdf(D, ndof)

        table_stats = np.zeros(1 , dtype=filaments_model_1f.dtype_stats)
        table_stats['kappa0_signif'] = kappa0_significance
        table_stats['kappa0_err_lo'] = kappa0_err_hi
        table_stats['kappa0_map'] = vmax_kappa0
        table_stats['kappa0_err_hi'] = kappa0_err_lo
        table_stats['radius_map'] = vmax_radius
        table_stats['radius_err_hi'] = radius_err_hi
        table_stats['radius_err_lo'] = radius_err_lo
        table_stats['chi2_red_null'] = chi2_red_null
        table_stats['chi2_red_max'] = chi2_red_max
        table_stats['chi2_D'] = chi2_D
        table_stats['chi2_LRT'] = chi2_LRT
        table_stats['chi2_red_D'] = chi2_D_red
        table_stats['chi2_red_LRT'] = chi2_LRT_red
        table_stats['chi2_null'] = chi2_null
        table_stats['chi2_max'] = chi2_max
        table_stats['id'] = id_pair
        table_stats['sigma_g'] = fitobj.sigma_g

        tabletools.saveTable(filename_results_pairs, table_stats, append=True)
        tabletools.savePickle(filename_results_prob,prob_post,append=True)
        
        grid_info = {}
        grid_info['kappa0_post_matrix'] = prob_post_kappa0
        grid_info['radius_post_matrix'] = prob_post_radius
        grid_info['kappa0_post'] = params[:,0]
        grid_info['radius_post'] = params[:,1]
        grid_info['grid_kappa0'] = grid_kappa0
        grid_info['grid_radius'] = grid_radius
        if id_pair == id_pair_first:
            tabletools.savePickle(filename_results_grid,grid_info)

        log.info('ML-ratio test: chi2_red_max=% 10.3f chi2_red_null=% 10.3f D_red=% 8.4e p-val_red=%1.5f' , chi2_red_max, chi2_red_null , chi2_D_red, chi2_LRT_red )
        log.info('ML-ratio test: chi2_max    =% 10.3f chi2_null    =% 10.3f D    =% 8.4e p-val    =%1.5f' , chi2_max, chi2_null , chi2_D, chi2_LRT )
        log.info('max %5.5f +%5.5f -%5.5f detection_significance=%5.2f', max_kappa0, kappa0_err_hi , kappa0_err_lo , kappa0_significance)

        if save_plots:


            pl.figure()
            pl.rcParams.update({'font.size': 2})
            pl.clf()
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(4, 6)

            pl.subplot(gs[0,0])
            plotstools.imshow_grid(params[:,0],params[:,1],prob_post,nx=n_grid,ny=n_grid)
            pl.plot( vmax_kappa0 , vmax_radius , 'rx' )
            pl.xlabel('kappa0')
            pl.ylabel('radius [Mpc]')

            pl.subplot(gs[0,1])
            plotstools.imshow_grid(params[:,0],params[:,1],log_post,nx=n_grid,ny=n_grid)
            pl.plot( vmax_kappa0 , vmax_radius , 'rx' )
            pl.xlabel('kappa0')
            pl.ylabel('radius [Mpc]')

            
            pl.subplot(gs[0,2])
            pl.plot(grid_kappa0 , prob_post_kappa0 )
            pl.axvline(x=max_kappa0 - kappa0_err_lo,linewidth=1, color='r')
            pl.axvline(x=max_kappa0 + kappa0_err_hi,linewidth=1, color='r')
            pl.xlabel('kappa0')

            pl.subplot(gs[0,3])
            pl.plot(grid_radius , prob_post_radius )
            pl.xlabel('radius [Mpc]')

            halo_marker_size = 10

           
            res1  = (fitobj.shear_g1 - best_model_g1) 
            res2  = (fitobj.shear_g2 - best_model_g2) 


            pl.subplot(gs[1,0:2])
            fitobj.plot_shears(fitobj.shear_g1,fitobj.shear_g2,limit_mask,unit='Mpc')
            pl.scatter(fitobj.halo1_u_mpc,fitobj.halo1_v_mpc,halo_marker_size,c='r')
            pl.scatter(fitobj.halo2_u_mpc,fitobj.halo2_v_mpc,halo_marker_size,c='r')
            pl.axhline(y= vmax_radius,linewidth=1, color='r')
            pl.axhline(y=-vmax_radius,linewidth=1, color='r')

            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])

            pl.subplot(gs[2,0:2])
            fitobj.plot_shears(best_model_g1,best_model_g2,limit_mask,unit='Mpc')
            pl.scatter(fitobj.halo1_u_mpc,fitobj.halo1_v_mpc,halo_marker_size,c='r')
            pl.scatter(fitobj.halo2_u_mpc,fitobj.halo2_v_mpc,halo_marker_size,c='r')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])

            pl.subplot(gs[3,0:2])
            fitobj.plot_shears(res1 , res2,limit_mask,unit='Mpc')
            pl.scatter(fitobj.halo1_u_mpc,fitobj.halo1_v_mpc,halo_marker_size,c='r')
            pl.scatter(fitobj.halo2_u_mpc,fitobj.halo2_v_mpc,halo_marker_size,c='r')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])


            scatter_size = 3.5
            maxg = max( [ max(abs(fitobj.shear_g1.flatten())) ,  max(abs(fitobj.shear_g2.flatten())) , max(abs(best_model_g1.flatten())) , max(abs(best_model_g1.flatten()))   ])

            pl.subplot(gs[1,2:4])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, fitobj.shear_g1 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            # plotstools.imshow_grid(fitobj.shear_u_mpc, fitobj.shear_v_mpc, fitobj.shear_g1)
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])
            
            pl.subplot(gs[2,2:4])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, best_model_g1 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])
            
            pl.subplot(gs[3,2:4])
            pl.axis('equal')
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, res1 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])

            pl.subplot(gs[1,4:])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, fitobj.shear_g2 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])
            
            pl.subplot(gs[2,4:])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, best_model_g2 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])
            
            pl.subplot(gs[3,4:])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, res2 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])

            title_str = 'id=%d R_pair=%.2f max=%1.2e (+%1.2e -%1.2e) nsig=%5.2f max_shear=%2.3f' % (id_pair, pairs_table[id_pair]['R_pair'], max_kappa0, kappa0_err_hi , kappa0_err_lo , kappa0_significance , maxg)
            title_str += '\nML-ratio test: chi2_red_max=%1.3f chi2_red_null=%1.3f D=%8.4e p-val=%1.3f' % (chi2_red_max, chi2_red_null , chi2_D, chi2_LRT)
            pl.suptitle(title_str)

            filename_fig = filename_fig = 'figs/result.%04d.%s.pdf' % (id_pair,filename_shears.replace('.fits',''))
            try:
                pl.savefig(filename_fig, dpi=300)
                log.info('saved %s' , filename_fig)
            except Exception , errmsg: 
                log.error('saving figure %s failed: %s' , filename_fig , errmsg)

            pl.clf()
            pl.close('all')

        # matplotlib leaks memory
        # print h.heap()

