import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools, plotstools, mathstools, scipy, scipy.stats
import numpy as np
import pylab as pl
# import matplotlib as mpl;
# from matplotlib import figure;
pl.rcParams['image.interpolation'] = 'nearest' ; 
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import filaments_model_1h
import filaments_model_1f

# print pl.get_backend()
# pl.rcParams['backend'] = 'agg'
# print pl.get_backend()

log = logging.getLogger("filam..fit") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

cospars = cosmology.cosmoparams()

prob_z = None

from guppy import hpy
h = hpy()

def fit_single_halo():



    id_pair = 7
    filename_shears = 'shears_bcc_g.%03d.fits' % id_pair
    filename_pairs = 'pairs_bcc.fits'
    filename_halo1 = 'pairs_bcc.halos1.fits'
    filename_halo2 = 'pairs_bcc.halos2.fits'

    pairs_table = tabletools.loadTable(filename_pairs)
    shears_info = tabletools.loadTable(filename_shears)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    true_M200 = np.log10(halo1_table['m200'][id_pair])
    true_M200 = np.log10(halo2_table['m200'][id_pair])
    log.info( 'halo1 M200 %5.2e',halo1_table['m200'][id_pair] )
    log.info( 'halo2 M200 %5.2e',halo2_table['m200'][id_pair] )
    log.info( 'halo1 conc %5.2f',halo1_table['r200'][id_pair]/halo1_table['rs'][id_pair]*1000.)
    log.info( 'halo2 conc %5.2f',halo2_table['r200'][id_pair]/halo2_table['rs'][id_pair]*1000.)


    fitobj = filaments_model_1h.modelfit()
    fitobj.get_bcc_pz()
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
    log_post , grid_M200, best_g1, best_g2, best_limit_mask = fitobj.run_gridsearch(M200_min=13.5,M200_max=15,M200_n=1000)
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


    import pdb;pdb.trace()


    
def fit_single_filament(save_plots=False,n_pairs=None):


    filename_pairs = 'pairs_bcc.fits'
    filename_halo1 = 'pairs_bcc.halos1.fits'
    filename_halo2 = 'pairs_bcc.halos2.fits'
    filaname_results = 'results_1f_bcc.fits'
    filename_shears = 'shears_bcc_g.fits'

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    # get prob_z
    fitobj = filaments_model_1f.modelfit()
    fitobj.get_bcc_pz()
    prob_z = fitobj.prob_z


    if n_pairs==None:
        n_pairs = len(pairs_table)

    pairs_table_use = pairs_table[:n_pairs]

    kappa0_signif = np.zeros(n_pairs)
    kappa0_map = np.zeros(n_pairs)
    kappa0_err_hi = np.zeros(n_pairs)
    kappa0_err_lo = np.zeros(n_pairs)
    radius_map = np.zeros(n_pairs)
    radius_err_hi = np.zeros(n_pairs)
    radius_err_lo = np.zeros(n_pairs)
    chi2_red_null = np.zeros(n_pairs)
    chi2_red_max = np.zeros(n_pairs)
    chi2_D = np.zeros(n_pairs)
    chi2_LRT = np.zeros(n_pairs)

    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='kappa0_signif', arr=kappa0_signif , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='kappa0_map',    arr=kappa0_map    , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='kappa0_err_hi', arr=kappa0_err_hi , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='kappa0_err_lo', arr=kappa0_err_lo , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='radius_map',    arr=radius_map    , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='radius_err_hi', arr=radius_err_hi , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='radius_err_lo', arr=radius_err_lo , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='chi2_red_null', arr=chi2_red_null , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='chi2_red_max',  arr=chi2_red_max  , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='chi2_D',        arr=chi2_D        , dtype='f8')
    pairs_table_use=tabletools.appendColumn(rec=pairs_table_use,name='chi2_LRT',      arr=chi2_LRT      , dtype='f8')

   
    for id_pair in range(args.first,args.last):


        # now we use that
        # filename_shears = 'shears_bcc_g.%03d.fits' % id_pair      
        # shears_info = tabletools.loadTable(filename_shears)

        # later use HDUs
        shears_info = tabletools.loadTable(filename_shears,hdu=id_pair+1)
        log.info('--------- pair %d with %d shears--------' , id_pair , len(shears_info)) 

        if len(shears_info) > 50000:
            log.warning('buggy pair, n_shears=%d , skipping' , len(shears_info))
            continue


        true_M200 = np.log10(halo1_table['m200'][id_pair])
        true_M200 = np.log10(halo2_table['m200'][id_pair])

        halo1_conc = halo1_table['r200'][id_pair]/halo1_table['rs'][id_pair]*1000.
        halo2_conc = halo2_table['r200'][id_pair]/halo2_table['rs'][id_pair]*1000.

        log.info( 'M200=[ %1.2e , %1.2e ] , conc=[ %1.2f , %1.2f ]',halo1_table['m200'][id_pair] , halo2_table['m200'][id_pair] , halo1_conc , halo2_conc)

        fitobj = filaments_model_1f.modelfit()
        fitobj.prob_z = prob_z
        fitobj.sigma_g =  0.001
        fitobj.shear_g1 =  shears_info['g1'] + np.random.randn(len(shears_info['g1']))*fitobj.sigma_g
        fitobj.shear_g2 =  shears_info['g2'] + np.random.randn(len(shears_info['g2']))*fitobj.sigma_g
        fitobj.shear_u_arcmin =  shears_info['u_arcmin']
        fitobj.shear_v_arcmin =  shears_info['v_arcmin']
        
        fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
        fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
        fitobj.halo1_z =  pairs_table['z'][id_pair]
        fitobj.halo1_M200 = halo1_table['m200'][id_pair]
        fitobj.halo1_conc = halo1_conc

        fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
        fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
        fitobj.halo2_z =  pairs_table['z'][id_pair]
        fitobj.halo2_M200 = halo2_table['m200'][id_pair]
        fitobj.halo2_conc = halo2_conc

        fitobj.parameters[0]['box']['min'] = 0.
        fitobj.parameters[0]['box']['max'] = 0.05
        fitobj.parameters[1]['box']['min'] = 0.001
        fitobj.parameters[1]['box']['max'] = 15
        
        # fitobj.plot_shears_mag(fitobj.shear_g1,fitobj.shear_g2)
        # pl.show()
        # fitobj.save_all_models=False
        log.info('running grid search')
        n_grid=150
        log_post , params, grid_kappa0, grid_radius = fitobj.run_gridsearch(n_grid=n_grid)

        # get the normalised PDF and use the same normalisation on the log
        prob_post , _ , _ , _ = get_normalisation(log_post)
        # get the maximum likelihood solution
        vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_kappa0 , vmax_radius = fitobj.get_grid_max(log_post,params)
        # get the marginals
        prob_post_matrix = np.reshape(  prob_post , [n_grid, n_grid] )    
        prob_post_kappa0 = prob_post_matrix.sum(axis=1)
        prob_post_radius = prob_post_matrix.sum(axis=0)
        # get confidence intervals on kappa0
        max_kappa0 , kappa0_err_hi , kappa0_err_lo = mathstools.estimate_confidence_interval(grid_kappa0 , prob_post_kappa0)
        max_radius , radius_err_hi , radius_err_lo = mathstools.estimate_confidence_interval(grid_kappa0 , prob_post_kappa0)
        err_use = np.mean([kappa0_err_hi,kappa0_err_lo])
        kappa0_significance = max_kappa0/err_use
        # get the likelihood test
        v_reducing_null = len(fitobj.shear_u_arcmin) - 1
        v_reducing_mod = len(fitobj.shear_u_arcmin) - 2 - 1
        chi2_red_null = fitobj.null_log_likelihood() / v_reducing_null
        chi2_red_max = vmax_post / v_reducing_mod
        chi2_D = 2*(chi2_red_max - chi2_red_null)
        ndof=2
        chi2_LRT = 1. - scipy.stats.chi2.cdf(chi2_D, ndof)
        # likelihood_ratio_test = scipy.stats.chi2.pdf(D, ndof)

        pairs_table_use['kappa0_signif'][id_pair] = kappa0_significance
        pairs_table_use['kappa0_err_lo'][id_pair] = vmax_kappa0
        pairs_table_use['kappa0_map'][id_pair] = kappa0_err_hi
        pairs_table_use['kappa0_err_hi'][id_pair] = kappa0_err_lo
        pairs_table_use['radius_map'][id_pair] = vmax_radius
        pairs_table_use['radius_err_hi'][id_pair] = radius_err_hi
        pairs_table_use['radius_err_lo'][id_pair] = radius_err_lo
        pairs_table_use['chi2_red_null'][id_pair] = chi2_red_null
        pairs_table_use['chi2_red_max'][id_pair] = chi2_red_max
        pairs_table_use['chi2_D'][id_pair] = chi2_D
        pairs_table_use['chi2_LRT'][id_pair] = chi2_LRT
        tabletools.saveTable(filaname_results,pairs_table_use)

        log.info('ML-ratio test: chi2_red_max=%1.3f chi2_red_null=%1.3f D=%8.4e p-val=%1.3f' , chi2_red_max, chi2_red_null , chi2_D, chi2_LRT )
        log.info('max %5.5f +%5.5f -%5.5f detection_significance=%5.2f', max_kappa0, kappa0_err_hi , kappa0_err_lo , kappa0_significance)

        if save_plots:


            pl.figure()
            pl.rcParams.update({'font.size': 5})
            pl.clf()
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(4, 6)

            pl.subplot(gs[0,0])
            plotstools.imshow_grid(params[:,0],params[:,1],prob_post,nx=n_grid,ny=n_grid)
            pl.plot( vmax_kappa0 , vmax_radius , 'rx' )

            pl.xlabel('kappa0')
            pl.ylabel('radius')

            pl.subplot(gs[0,1])
            plotstools.imshow_grid(params[:,0],params[:,1],log_post,nx=n_grid,ny=n_grid)
            pl.plot( vmax_kappa0 , vmax_radius , 'rx' )
            pl.xlabel('kappa0')
            pl.ylabel('radius')

            
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
            fitobj.plot_shears(fitobj.shear_g1,fitobj.shear_g2,limit_mask)
            pl.scatter(fitobj.halo1_u_arcmin,fitobj.halo1_v_arcmin,halo_marker_size,c='r')
            pl.scatter(fitobj.halo2_u_arcmin,fitobj.halo2_v_arcmin,halo_marker_size,c='r')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_arcmin),max(fitobj.shear_u_arcmin)])
            pl.ylim([min(fitobj.shear_v_arcmin),max(fitobj.shear_v_arcmin)])

            pl.subplot(gs[2,0:2])
            fitobj.plot_shears(best_model_g1,best_model_g2,limit_mask)
            pl.scatter(fitobj.halo1_u_arcmin,fitobj.halo1_v_arcmin,halo_marker_size,c='r')
            pl.scatter(fitobj.halo2_u_arcmin,fitobj.halo2_v_arcmin,halo_marker_size,c='r')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_arcmin),max(fitobj.shear_u_arcmin)])
            pl.ylim([min(fitobj.shear_v_arcmin),max(fitobj.shear_v_arcmin)])

            pl.subplot(gs[3,0:2])
            fitobj.plot_shears(res1 , res2,limit_mask)
            pl.scatter(fitobj.halo1_u_arcmin,fitobj.halo1_v_arcmin,halo_marker_size,c='r')
            pl.scatter(fitobj.halo2_u_arcmin,fitobj.halo2_v_arcmin,halo_marker_size,c='r')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_arcmin),max(fitobj.shear_u_arcmin)])
            pl.ylim([min(fitobj.shear_v_arcmin),max(fitobj.shear_v_arcmin)])


            scatter_size = 3.5
            maxg = max( [ max(abs(fitobj.shear_g1.flatten())) ,  max(abs(fitobj.shear_g2.flatten())) , max(abs(best_model_g1.flatten())) , max(abs(best_model_g1.flatten()))   ])

            pl.subplot(gs[1,2:4])
            pl.scatter(fitobj.shear_u_arcmin, fitobj.shear_v_arcmin, scatter_size, fitobj.shear_g1 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_arcmin),max(fitobj.shear_u_arcmin)])
            pl.ylim([min(fitobj.shear_v_arcmin),max(fitobj.shear_v_arcmin)])
            
            pl.subplot(gs[2,2:4])
            pl.scatter(fitobj.shear_u_arcmin, fitobj.shear_v_arcmin, scatter_size, best_model_g1 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_arcmin),max(fitobj.shear_u_arcmin)])
            pl.ylim([min(fitobj.shear_v_arcmin),max(fitobj.shear_v_arcmin)])
            
            pl.subplot(gs[3,2:4])
            pl.axis('equal')
            pl.scatter(fitobj.shear_u_arcmin, fitobj.shear_v_arcmin, scatter_size, res1 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.xlim([min(fitobj.shear_u_arcmin),max(fitobj.shear_u_arcmin)])
            pl.ylim([min(fitobj.shear_v_arcmin),max(fitobj.shear_v_arcmin)])

            pl.subplot(gs[1,4:])
            pl.scatter(fitobj.shear_u_arcmin, fitobj.shear_v_arcmin, scatter_size, fitobj.shear_g2 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_arcmin),max(fitobj.shear_u_arcmin)])
            pl.ylim([min(fitobj.shear_v_arcmin),max(fitobj.shear_v_arcmin)])
            
            pl.subplot(gs[2,4:])
            pl.scatter(fitobj.shear_u_arcmin, fitobj.shear_v_arcmin, scatter_size, best_model_g2 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_arcmin),max(fitobj.shear_u_arcmin)])
            pl.ylim([min(fitobj.shear_v_arcmin),max(fitobj.shear_v_arcmin)])
            
            pl.subplot(gs[3,4:])
            pl.scatter(fitobj.shear_u_arcmin, fitobj.shear_v_arcmin, scatter_size, res2 , lw = 0 , vmax=maxg , vmin=-maxg , marker='s')
            pl.axis('equal')
            pl.xlim([min(fitobj.shear_u_arcmin),max(fitobj.shear_u_arcmin)])
            pl.ylim([min(fitobj.shear_v_arcmin),max(fitobj.shear_v_arcmin)])

            title_str = 'id=%d R_pair=%.2f max=%1.2e (+%1.2e -%1.2e) nsig=%5.2f max_shear=%2.3f' % (id_pair, pairs_table[id_pair]['R_pair'], max_kappa0, kappa0_err_hi , kappa0_err_lo , kappa0_significance , maxg)
            title_str += '\nML-ratio test: chi2_red_max=%1.3f chi2_red_null=%1.3f D=%8.4e p-val=%1.3f' % (chi2_red_max, chi2_red_null , chi2_D, chi2_LRT)
            pl.suptitle(title_str)

            filename_fig = filename_fig = 'figs/result.%04d.pdf' % id_pair
            try:
                pl.savefig(filename_fig, dpi=1000)
                log.info('saved %s' , filename_fig)
            except: 
                log.error('saving figure %s failed' , filename_fig )

            pl.clf()
            pl.close()

        print h.heap()




def get_normalisation(log_post):

    interm_norm = max(log_post.flatten())
    log_post = log_post - interm_norm
    prob_post = np.exp(log_post)
    prob_norm = np.sum(prob_post)
    prob_post = prob_post / prob_norm
    log_post  = np.log(prob_post)
    log_norm = np.log(prob_norm) + interm_norm
       
    return prob_post , log_post , prob_norm , log_norm


def main():

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    # parser.add_argument('-c', '--filename_config', default='test2.yaml',type=str, action='store', help='name of the yaml config file')
    parser.add_argument('-l', '--last', default=0,type=int, action='store', help='first pair to process')
    parser.add_argument('-f', '--first', default=0,type=int, action='store', help='first pair to process')
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

    filaments_model_1f.log = log
    # plotstools.log = log

    # grid_search(pair_info,shears_info)
    # test_model(shears_info,pair_info)

    # fit_single_halo()
    fit_single_filament(save_plots=True,n_pairs=3000)

def delete_module(modname, paranoid=None):
    from sys import modules
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(modname)
    these_symbols = dir(thismod)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[modname]
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    pass


main()