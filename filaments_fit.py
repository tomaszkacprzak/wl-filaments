import os
if os.environ['USER'] == 'ucabtok': os.environ['MPLCONFIGDIR']='.'
import matplotlib as mpl
if 'DISPLAY' not in os.environ:
    mpl.use('agg')
import os, yaml, argparse, sys, logging , pyfits, tabletools, cosmology, filaments_tools, plotstools, mathstools, scipy, scipy.stats, time
import numpy as np
import pylab as pl
print 'using matplotlib backend' , pl.get_backend()
# import matplotlib as mpl;
# from matplotlib import figure;
pl.rcParams['image.interpolation'] = 'nearest' ; 
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import cPickle as pickle
import filaments_model_1h
import filaments_model_1f
import filaments_model_2hf
import shutil
try:
    from pyqt_fit import kde
except: 
    print 'importing pyqt_fit failed'



log = logging.getLogger("filam..fit") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

cospars = cosmology.cosmoparams()

prob_z = None

N_BURNIN = 1000

def fit_2hf():

    filename_pairs =  config['filename_pairs']                                   # pairs_bcc.fits'
    filename_halo1 =  config['filename_pairs'].replace('.fits' , '.halos1.fits') # pairs_bcc.halos1.fits'
    filename_halo2 =  config['filename_pairs'].replace('.fits' , '.halos2.fits') # pairs_bcc.halos2.fits'
    filename_shears = config['filename_shears']                                  # args.filename_shears 

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    if args.first == -1: 
        id_pair_first = 0 ; 
    else: 
        id_pair_first = args.first
    
    id_pair_last = args.first + args.num 
    n_pairs_available = len(pairs_table)

    if config['mode']!='selftest':
        if id_pair_first > len(pairs_table):
            raise Exception('id_pair_first=%d greater than number of pairs=%d' % (id_pair_first, len(pairs_table) ))

    if id_pair_last > (n_pairs_available-1):
        id_pair_last = n_pairs_available

    if config['mode']=='selftest':
        id_pair_first = args.first
        id_pair_last = args.first + args.num 


    log.info('running on pairs from %d to %d' , id_pair_first , id_pair_last)

    filename_results_prob = 'results.prob.%04d.%04d.' % (id_pair_first, id_pair_last) +   os.path.basename(filename_shears).replace('.fits','.pp2')
    filename_results_grid = 'results.grid.' +   os.path.basename(filename_shears).replace('.fits','.pp2')
    filename_results_chain = 'results.chain.%04d.%04d.' % (id_pair_first, id_pair_last) + os.path.basename(filename_shears).replace('.fits','.pp2')
    filename_results_pairs = 'results.stats.%04d.%04d.' % (id_pair_first, id_pair_last) + os.path.basename(filename_shears).replace('.fits','.cat')
    if os.path.isfile(filename_results_prob):
        os.remove(filename_results_prob)
        log.warning('overwriting file %s ' , filename_results_prob)
    if os.path.isfile(filename_results_pairs):
        os.remove(filename_results_pairs)
        log.warning('overwriting file %s ' , filename_results_pairs)
    if os.path.isfile(filename_results_grid):
        os.remove(filename_results_grid)
        log.warning('overwriting file %s ' , filename_results_grid)
    if os.path.isfile(filename_results_chain):
        os.remove(filename_results_chain)
        log.warning('overwriting file %s ' , filename_results_chain)


    if config['get_confidence_intervals']:
        tabletools.writeHeader(filename_results_pairs,filaments_model_2hf.dtype_stats)
    

    # get prob_z
    fitobj = filaments_model_2hf.modelfit()
    fitobj.get_bcc_pz(config['filename_pz'])
    prob_z = fitobj.prob_z
    sigma_ell = fitobj.sigma_ell
    grid_z_centers = fitobj.grid_z_centers
    grid_z_edges = fitobj.grid_z_edges

    # empty container list for probability measurements
    # table_stats = np.zeros(len(range(id_pair_first,id_pair_last)) , dtype=dtype_stats)

    for id_pair in range(id_pair_first,id_pair_last):

    	id_shear = pairs_table[id_pair]['ipair']
        log.info('--------- pair %d shear %d --------' , id_pair, id_shear) 
        # now we use that
        if config['mode']=='selftest':
            id_pair_in_catalog = 0
            if '.fits' in filename_shears:
                shears_info = tabletools.loadTable(filename_shears,hdu=1)
            elif '.pp2' in filename_shears:
                shears_info = tabletools.loadPickle(filename_shears,pos=0)
            log.info('selftest mode - using HDU=1 and adding noise')
        else:
            id_pair_in_catalog = id_pair
            if '.fits' in filename_shears:
                shears_info = tabletools.loadTable(filename_shears,hdu=id_shear+1)
            elif '.pp2' in filename_shears:
                shears_info = tabletools.loadPickle(filename_shears,pos=id_shear)

        log.info('using %d shears' , len(shears_info) )

        if len(shears_info) > 500000:
            log.warning('buggy pair, n_shears=%d , skipping' , len(shears_info))
            result_dict = {'id' : id_pair}
            tabletools.savePickle(filename_results_prob,prob_result,append=True)
            continue
       

        fitobj = filaments_model_2hf.modelfit()
        fitobj.kappa_is_K = config['kappa_is_K']
        fitobj.prob_z = prob_z
        fitobj.grid_z_centers = grid_z_centers
        fitobj.grid_z_edges = grid_z_edges
        fitobj.sigma_ell = sigma_ell
        fitobj.shear_u_arcmin =  shears_info['u_arcmin']
        fitobj.shear_v_arcmin =  shears_info['v_arcmin']
        fitobj.shear_u_mpc =  shears_info['u_mpc']
        fitobj.shear_v_mpc =  shears_info['v_mpc']
        fitobj.shear_g1 =  shears_info['g1']
        fitobj.shear_g2 =  shears_info['g2']
        fitobj.shear_w =  shears_info['weight']

        # choose a method to add and account for noise
        if config['sigma_method'] == 'add':
            sigma_g_add =  config['sigma_add']
            fitobj.shear_g1 =  shears_info['g1'] + np.random.randn(len(shears_info['g1']))*sigma_g_add
            fitobj.shear_g2 =  shears_info['g2'] + np.random.randn(len(shears_info['g2']))*sigma_g_add
            fitobj.sigma_g =  np.std(fitobj.shear_g2,ddof=1)
            fitobj.sigma_ell = fitobj.sigma_g
            fitobj.inv_sq_sigma_g = 1./fitobj.sigma_g**2
            log.info('added noise with level %f , using sigma_g=%2.5f' , sigma_g_add, fitobj.sigma_g)
        elif config['sigma_method'] == 'orig':
            fitobj.shear_n_gals = shears_info['n_gals']
            fitobj.inv_sq_sigma_g = fitobj.shear_w
            log.info('using different sigma_g per pixel mean(inv_sq_sigma_g)=%2.5f len(inv_sq_sigma_g)=%d' , np.mean(fitobj.inv_sq_sigma_g) , len(fitobj.inv_sq_sigma_g))
                    
        fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair_in_catalog]
        fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair_in_catalog]
        fitobj.halo1_u_mpc =  pairs_table['u1_mpc'][id_pair_in_catalog]
        fitobj.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair_in_catalog]
        fitobj.halo1_z =  pairs_table['z'][id_pair_in_catalog]
        # fitobj.halo1_M200 = halo1_table['m200'][id_pair_in_catalog]
        # fitobj.halo1_conc = halo1_conc

        fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair_in_catalog]
        fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair_in_catalog]
        fitobj.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair_in_catalog]
        fitobj.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair_in_catalog]
        fitobj.halo2_z =  pairs_table['z'][id_pair_in_catalog]
        # fitobj.halo2_M200 = halo2_table['m200'][id_pair_in_catalog]
        # fitobj.halo2_conc = halo2_conc

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

        # fitobj.plot_shears_mag(fitobj.shear_g1,fitobj.shear_g2)
        # pl.show()
        # fitobj.save_all_models=False

        if config['optimization_mode'] == 'mcmc':
            log.info('running sampling')
            fitobj.n_walkers = config['n_walkers']
            fitobj.n_samples = config['n_samples']
            fitobj.n_grid = config['n_grid']
            fitobj.run_mcmc()
            log_post = fitobj.sampler.flatlnprobability
            params = fitobj.sampler.flatchain

            n_grid = fitobj.n_grid                      

            list_prob_marg = []
            list_params_marg = []

            for di in config['get_marginals_for_params']:

                # list_params_marg.append()
                
                if N_BURNIN < len(fitobj.sampler.flatchain):
                    chain = fitobj.sampler.flatchain[N_BURNIN:,di]
                else:
                    chain = fitobj.sampler.flatchain[:,di]
                
                # kde_est = kde.KDE1D(chain, lower=fitobj.parameters[di]['box']['min'] , upper=fitobj.parameters[di]['box']['max'] , method='linear_combination')                          
                # kde_est = kde.KDE1D(chain, lower=fitobj.parameters[di]['box']['min'] , upper=fitobj.parameters[di]['box']['max'] , method='reflexion')                          
                # marg_prob =mathstools.get_func_split(grid=list_params_marg[di],func=kde_est)
                               
                from scipy.stats.kde import gaussian_kde
                kde_est = gaussian_kde(chain)
                xxs = np.linspace(fitobj.parameters[di]['box']['min'],fitobj.parameters[di]['box']['max'],n_grid)
                yys = kde_est(xxs)
                marg_prob = yys
                list_params_marg.append(xxs)
                list_prob_marg.append(yys)
                log.info('param %d KDE bandwidth=%2.3f normalisation=%f', di, kde_est.factor , np.sum(marg_prob))

                # from sklearn import mixture
                # gmm=mixture.GMM(n_components=40)
                # gmm.fit(chain)
                # xxs = np.linspace(fitobj.parameters[di]['box']['min'],fitobj.parameters[di]['box']['max'],n_grid)
                # yys = np.exp(gmm.score(xxs))
                # list_params_marg.append(xxs)
                # list_prob_marg.append(yys)

                # pl.figure()
                # pl.plot(list_params_marg[di] , list_prob_marg[di], 'x')
                # pl.hist(chain,bins=list_params_marg[di], normed=True)
                # pl.xlabel(str(di))
                # pl.show()

            # get kappa - radius distribution
            if N_BURNIN < len(fitobj.sampler.flatchain):
                    chain = fitobj.sampler.flatchain[N_BURNIN:,:2]
            else:
                chain = fitobj.sampler.flatchain[:,:2]
            from scipy.stats.kde import gaussian_kde
            kde_est = gaussian_kde(chain.T)

            grid1 = np.linspace(fitobj.parameters[0]['box']['min'],fitobj.parameters[0]['box']['max'],config['n_grid_2D'])
            grid2 = np.linspace(fitobj.parameters[1]['box']['min'],fitobj.parameters[1]['box']['max'],config['n_grid_2D'])
            xx,yy=np.meshgrid(grid1,grid2); 
            grid12=np.concatenate([xx.flatten()[:,None],yy.flatten()[:,None]],axis=1)
            marg_prob = kde_est(grid12.T)   
            log.info('params kappa0 radius KDE bandwidth=%2.3f normalisation=%f', kde_est.factor , np.sum(marg_prob))



            grids = list_params_marg

            vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_params = fitobj.get_samples_max(log_post,params)
            chain_result = {}
            chain_result['flatlnprobability'] = fitobj.sampler.flatlnprobability.astype(np.float32),
            chain_result['flatchain'] = fitobj.sampler.flatchain.astype(np.float32),
            chain_result['list_prob_marg'] = list_prob_marg
            chain_result['prob_kappa0_radius'] = marg_prob
            chain_result['id'] = id_pair
            chain_params = {}
            chain_params['list_params_marg'] = list_params_marg
            chain_params['kappa0_radius_grid'] = grid12
            tabletools.savePickle(filename_results_chain, chain_result ,append=True)
            tabletools.savePickle(filename_results_grid, chain_params)


        elif config['optimization_mode'] == 'gridsearch':
            log.info('running grid search')

            log_post , params, grids = fitobj.run_gridsearch()
            tabletools.savePickle(filename_results_prob,log_post.astype(np.float32),append=True)

            # get the normalised PDF and use the same normalisation on the log
            prob_post , _ , _ , _ = mathstools.get_normalisation(log_post)
            # # get the maximum likelihood solution
            # vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_params = fitobj.get_grid_max(log_post,params)
            # # get the marginals
            # list_prob_marg, list_params_marg = mathstools.get_marginals(params,prob_post)

            
            if id_pair == id_pair_first:
                grid_info = {}
                grid_info['grid_kappa0'] = grids[0]
                grid_info['grid_radius'] = grids[1]
                grid_info['grid_h1M200'] = grids[2]
                grid_info['grid_h2M200'] = grids[3]
                grid_info['post_kappa0'] = params[0]
                grid_info['post_radius'] = params[1]
                grid_info['post_h1M200'] = params[2]
                grid_info['post_h2M200'] = params[3]
                tabletools.savePickle(filename_results_grid,grid_info)

            # pl.subplot(2,2,1)
            # pl.plot(list_params_marg[0], list_prob_marg[0])
            # pl.subplot(2,2,2)
            # pl.plot(list_params_marg[1], list_prob_marg[1])
            # pl.subplot(2,2,3)
            # pl.plot(list_params_marg[2], list_prob_marg[2])
            # pl.subplot(2,2,4)
            # pl.plot(list_params_marg[3], list_prob_marg[3])
               
            # list_prob_marg, list_params_marg = mathstools.get_marginals_2d(params,prob_post)   
            # ia=0
            # pl.figure()
            # for i in range(4):
            #     for j in range(4):
            #         ia+=1
            #         pl.subplot(4,4,ia)
            #         pl.imshow(list_prob_marg[i][j].T,extent=[ min(list_params_marg[i][j][0]) , max(list_params_marg[i][j][0]) , min(list_params_marg[i][j][1]) , max(list_params_marg[i][j][1]) ] , aspect='auto')

        # get confidence intervals 
        if config['get_confidence_intervals']:

            max_kappa0 , kappa0_err_hi , kappa0_err_lo = mathstools.estimate_confidence_interval(grids[0] , list_prob_marg[0])
            max_radius , radius_err_hi , radius_err_lo = mathstools.estimate_confidence_interval(grids[1] , list_prob_marg[1])
            max_h1M200 , h1M200_err_hi , h1M200_err_lo = mathstools.estimate_confidence_interval(grids[2] , list_prob_marg[2])
            max_h2M200 , h2M200_err_hi , h2M200_err_lo = mathstools.estimate_confidence_interval(grids[3] , list_prob_marg[3])

            err_use = np.mean([kappa0_err_hi,kappa0_err_lo])
            kappa0_significance = max_kappa0/err_use
            # get the likelihood test
            v_reducing_null = len(fitobj.shear_u_arcmin) - 1
            v_reducing_mod = len(fitobj.shear_u_arcmin) - 2 - 1
            chi2_null = fitobj.null_log_likelihood(max_h1M200,max_h2M200)
            chi2_max = vmax_post
            chi2_red_null = chi2_null / v_reducing_null
            chi2_red_max = chi2_max / v_reducing_mod
            chi2_D = 2*(chi2_max - chi2_null)
            chi2_D_red = 2*(chi2_red_max - chi2_red_null)
            ndof=2
            chi2_LRT_red = 1. - scipy.stats.chi2.cdf(chi2_D_red, ndof)
            chi2_LRT = 1. - scipy.stats.chi2.cdf(chi2_D, ndof)
            # likelihood_ratio_test = scipy.stats.chi2.pdf(D, ndof)

            table_stats = np.zeros(1 , dtype=filaments_model_2hf.dtype_stats)
            table_stats['kappa0_signif'] = kappa0_significance
            table_stats['kappa0_map'] = vmax_params[0]
            table_stats['kappa0_err_lo'] = kappa0_err_hi
            table_stats['kappa0_err_hi'] = kappa0_err_lo
            table_stats['radius_map'] = vmax_params[1]
            table_stats['radius_err_hi'] = radius_err_hi
            table_stats['radius_err_lo'] = radius_err_lo
            table_stats['h1M200_map'] = vmax_params[2]
            table_stats['h1M200_err_hi'] = h1M200_err_hi
            table_stats['h1M200_err_lo'] = h1M200_err_lo
            table_stats['h2M200_map'] = vmax_params[3]
            table_stats['h2M200_err_hi'] = h2M200_err_hi
            table_stats['h2M200_err_lo'] = h2M200_err_lo
            table_stats['chi2_red_null'] = chi2_red_null
            table_stats['chi2_red_max'] = chi2_red_max
            table_stats['chi2_D'] = chi2_D
            table_stats['chi2_LRT'] = chi2_LRT
            table_stats['chi2_red_D'] = chi2_D_red
            table_stats['chi2_red_LRT'] = chi2_LRT_red
            table_stats['chi2_null'] = chi2_null
            table_stats['chi2_max'] = chi2_max
            table_stats['id'] = id_pair
            table_stats['sigma_g'] = fitobj.sigma_ell

            tabletools.saveTable(filename_results_pairs, table_stats, append=True)
        
            # log.info('ML-ratio test: chi2_red_max=% 10.3f chi2_red_null=% 10.3f D_red=% 8.4e p-val_red=%1.5f' , chi2_red_max, chi2_red_null , chi2_D_red, chi2_LRT_red )
            log.info('ML-ratio test: chi2_max    =% 10.3f chi2_null    =% 10.3f D    =% 8.4e p-val    =%1.5f' , chi2_max, chi2_null , chi2_D, chi2_LRT )
            log.info('max %5.5f +%5.5f -%5.5f detection_significance=%5.2f', max_kappa0, kappa0_err_hi , kappa0_err_lo , kappa0_significance)

        if config['save_dist_plots']:
            
            pl.rcParams.update({'font.size': 8})
            pl.figure()
            pl.clf()

            # get plot of distributions
            if config['optimization_mode'] == 'mcmc':

                if N_BURNIN < len(fitobj.sampler.flatchain):
                    chain = fitobj.sampler.flatchain[N_BURNIN:,:]
                else:
                    chain = fitobj.sampler.flatchain

                bins = [ np.linspace(0.,fitobj.parameters[0]['box']['max'],100) ,
                         np.linspace(0.,fitobj.parameters[1]['box']['max'],100) ,
                         np.linspace(fitobj.parameters[2]['box']['min'],fitobj.parameters[2]['box']['max'],100) ,
                         np.linspace(fitobj.parameters[3]['box']['min'],fitobj.parameters[3]['box']['max'],100) ]

                plotstools.plot_dist(chain,bins=bins,labels=[r'$\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}$' , 'radius Mpc' , r'halo1_M200 $M_{*}$' , 'halo2_M200 $M_{*}$'])

            elif config['optimization_mode'] == 'gridsearch':
                                  
                
                 plotstools.plot_dist_grid(params,prob_post,labels=[r'$\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}$' , 'radius Mpc' , r'halo1_M200 $M_{*}$' , 'halo2_M200 $M_{*}$'])

            # get title
            if config['get_confidence_intervals']:
                title_str = 'id=%d R_pair=%.2f max=%1.2e (+%1.2e -%1.2e) nsig=%5.2f' % (id_pair, pairs_table[id_pair_in_catalog]['R_pair'], max_kappa0, kappa0_err_hi , kappa0_err_lo , kappa0_significance)
                title_str += '\nML-ratio test: chi2_red_max=%1.3f chi2_red_null=%1.3f D=%8.4e p-val=%1.3f' % (chi2_red_max, chi2_red_null , chi2_D, chi2_LRT)
            else:
                title_str = 'id=%d R_pair=%.2f' % (id_pair, pairs_table[id_pair_in_catalog]['R_pair'])
            pl.suptitle(title_str)

            # save figure
            filename_fig = 'figs/result.%04d.%s.dist.pdf' % (id_pair,os.path.basename(filename_shears).replace('.fits',''))
            try:
                pl.savefig(filename_fig, dpi=300)
                log.info('saved %s' , filename_fig)
            except Exception , errmsg: 
                log.error('saving figure %s failed: %s' , filename_fig , errmsg)

            pl.clf()
            pl.close('all')

        if config['save_model_plots']:

            pl.figure()

            import matplotlib.gridspec as gridspec          
            gs = gridspec.GridSpec(3, 6)

            halo_marker_size = 10
         
            res1  = (fitobj.shear_g1 - best_model_g1) 
            res2  = (fitobj.shear_g2 - best_model_g2) 


            pl.subplot(gs[0,0:2])
            fitobj.plot_shears(fitobj.shear_g1,fitobj.shear_g2,limit_mask,unit='Mpc')
            pl.scatter(fitobj.halo1_u_mpc,fitobj.halo1_v_mpc,halo_marker_size,c='r')
            pl.scatter(fitobj.halo2_u_mpc,fitobj.halo2_v_mpc,halo_marker_size,c='r')
            pl.axhline(y= vmax_params[1],linewidth=1, color='r')
            pl.axhline(y=-vmax_params[1],linewidth=1, color='r')

            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])

            pl.subplot(gs[1,0:2])
            fitobj.plot_shears(best_model_g1,best_model_g2,limit_mask,unit='Mpc')
            pl.scatter(fitobj.halo1_u_mpc,fitobj.halo1_v_mpc,halo_marker_size,c='r')
            pl.scatter(fitobj.halo2_u_mpc,fitobj.halo2_v_mpc,halo_marker_size,c='r')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])

            pl.subplot(gs[2,0:2])
            fitobj.plot_shears(res1 , res2,limit_mask,unit='Mpc')
            pl.scatter(fitobj.halo1_u_mpc,fitobj.halo1_v_mpc,halo_marker_size,c='r')
            pl.scatter(fitobj.halo2_u_mpc,fitobj.halo2_v_mpc,halo_marker_size,c='r')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])


            scatter_size = 3.5
            maxg = max( [ max(abs(fitobj.shear_g1.flatten())) ,  max(abs(fitobj.shear_g2.flatten())) , max(abs(best_model_g1.flatten())) , max(abs(best_model_g1.flatten()))   ])

            pl.subplot(gs[0,2:4])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, fitobj.shear_g1 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            # plotstools.imshow_grid(fitobj.shear_u_mpc, fitobj.shear_v_mpc, fitobj.shear_g1)
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])
            
            pl.subplot(gs[1,2:4])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, best_model_g1 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])
            
            pl.subplot(gs[2,2:4])
            pl.axis('equal')
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, res1 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])

            pl.subplot(gs[0,4:])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, fitobj.shear_g2 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])
            
            pl.subplot(gs[1,4:])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, best_model_g2 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])
            
            pl.subplot(gs[2,4:])
            pl.scatter(fitobj.shear_u_mpc, fitobj.shear_v_mpc, scatter_size, res2 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_mpc),max(fitobj.shear_u_mpc)])
            pl.ylim([min(fitobj.shear_v_mpc),max(fitobj.shear_v_mpc)])

            if config['get_confidence_intervals']:
                title_str = 'id=%d R_pair=%.2f max=%1.2e (+%1.2e -%1.2e) nsig=%5.2f max_shear=%2.3f' % (id_pair, pairs_table[id_pair_in_catalog]['R_pair'], max_kappa0, kappa0_err_hi , kappa0_err_lo , kappa0_significance , maxg)
                title_str += '\nML-ratio test: chi2_red_max=%1.3f chi2_red_null=%1.3f D=%8.4e p-val=%1.3f' % (chi2_red_max, chi2_red_null , chi2_D, chi2_LRT)
            else:
                title_str = 'id=%d R_pair=%.2f' % (id_pair, pairs_table[id_pair_in_catalog]['R_pair'])
                pl.suptitle(title_str)

            filename_fig = filename_fig = 'figs/result.%04d.%s.model.pdf' % (id_pair,os.path.basename(filename_shears).replace('.fits',''))
            try:
                pl.savefig(filename_fig, dpi=300)
                log.info('saved %s' , filename_fig)
            except Exception , errmsg: 
                log.error('saving figure %s failed: %s' , filename_fig , errmsg)

            pl.clf()
            pl.close('all')




def main():

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    # parser.add_argument('-c', '--filename_config', default='test2.yaml',type=str, action='store', help='name of the yaml config file')
    parser.add_argument('-f', '--first', default=0,type=int, action='store', help='first pair to process')
    parser.add_argument('-n', '--num', default=1,type=int, action='store', help='number of pairs to process')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    # parser.add_argument('-fg', '--filename_shears', type=str, default='shears_bcc_g.fits' , action='store', help='filename of file containing shears in binned format')
    # parser.add_argument('-d', '--dry', default=False,  action='store_true', help='Dry run, dont generate data')

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
    filaments_tools.config = config


    filaments_model_1f.log = log
    # plotstools.log = log

    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    # grid_search(pair_info,shears_info)
    # test_model(shears_info,pair_info)

    # fit_single_halo()
    # fit_single_filament(save_plots=args.save_plots)
    fit_2hf()
    # process_results()
    # analyse_results()
    # analyse_stats()

    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


main()
