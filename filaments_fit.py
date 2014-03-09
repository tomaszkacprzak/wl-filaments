import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools, plotstools
import numpy as np
import pylab as pl; pl.rcParams['image.interpolation'] = 'nearest'
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import filaments_model_1h
import filaments_model_1f

log = logging.getLogger("filam..fit") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False


cospars = cosmology.cosmoparams()

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


    import pdb;pdb.set_trace()

def estimate_confidence_interval(par_orig,pdf_orig):
    import scipy
    import scipy.interpolate


    # upsample PDF
    n_upsample = 10000
    f = scipy.interpolate.interp1d(par_orig,pdf_orig)
    par = np.linspace(min(par_orig),max(par_orig),n_upsample) 
    pdf = f(par)

    sig = 1
    confidence_level = scipy.special.erf( float(sig) / np.sqrt(2.) )

    pdf_norm = sum(pdf.flatten())
    pdf = pdf/pdf_norm

    max_pdf = max(pdf.flatten())
    min_pdf = 0.

    max_par = par[pdf.argmax()]

    list_levels , _ = plotstools.get_sigma_contours_levels(pdf,list_sigmas=[1])
    sig1_level = list_levels[0]

    diff = abs(pdf-sig1_level)
    ix1,ix2 = diff.argsort()[:2]
    par_x1 , par_x2 = par[ix1] , par[ix2]
    sig_point_lo = min([par_x1,par_x2])
    sig_point_hi = max([par_x1,par_x2])

    pl.figure()
    pl.plot(par,pdf,'x-')
    pl.axvline(x=sig_point_lo,linewidth=1, color='r')
    pl.axvline(x=sig_point_hi,linewidth=1, color='r')


    err_hi = sig_point_hi - max_par
    err_lo = max_par - sig_point_lo 
      
    log.debug('max %5.5f +%5.5f -%5.5f', max_par, err_hi , err_lo)

    return  max_par , err_hi , err_lo




    
def fit_single_filament():

    id_pair = 7
    # fitobj.parameters[0]['box']['min'] = 0.01
    # fitobj.parameters[0]['box']['max'] = 0.03
    # fitobj.parameters[1]['box']['min'] = 0.1
    # fitobj.parameters[1]['box']['max'] = 2.5


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

    halo1_conc = halo1_table['r200'][id_pair]/halo1_table['rs'][id_pair]*1000.
    halo2_conc = halo2_table['r200'][id_pair]/halo2_table['rs'][id_pair]*1000.

    log.info( 'halo1 M200 %5.2e',halo1_table['m200'][id_pair] )
    log.info( 'halo2 M200 %5.2e',halo2_table['m200'][id_pair] )
    log.info( 'halo1 conc %5.2f',halo1_conc)
    log.info( 'halo2 conc %5.2f',halo2_conc)

    fitobj = filaments_model_1f.modelfit()
    fitobj.get_bcc_pz()
    fitobj.sigma_g =  0.2
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

    fitobj.parameters[0]['box']['min'] = 0.0001
    fitobj.parameters[0]['box']['max'] = 0.3
    fitobj.parameters[1]['box']['min'] = 0.0001
    fitobj.parameters[1]['box']['max'] = 10
    
    # fitobj.plot_shears_mag(fitobj.shear_g1,fitobj.shear_g2)
    # pl.show()
    # fitobj.save_all_models=False
    log.info('running grid search')
    n_grid=100
    log_post , params, grid_kappa0, grid_radius = fitobj.run_gridsearch(n_grid=n_grid)
    vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_kappa0 , vmax_radius = fitobj.get_grid_max(log_post,params)


    scatter_size=10

    pl.figure()
    pl.subplot(1,2,1)
    prob_post = get_post_from_log(log_post)
    pl.scatter( params[:,0] , params[:,1] , scatter_size , log_post , lw=0)
    pl.colorbar()
    pl.subplot(1,2,2)
    pl.scatter( params[:,0] , params[:,1] , scatter_size , prob_post , lw=0)
    pl.plot( vmax_kappa0 , vmax_radius , 'ro' )
    pl.colorbar()
    filename_fig = 'post.png'
    pl.savefig(filename_fig, dpi=1000)

    # grid_kappa0_matrix = np.reshape(  grid_kappa0 , [ n_grid, n_grid] ) 
    # grid_radius_matrix = np.reshape(  grid_radius , [ n_grid, n_grid] )
    prob_post_matrix = np.reshape(  prob_post , [n_grid, n_grid] )    

    prob_post_kappa0 = prob_post_matrix.sum(axis=1)
    prob_post_radius = prob_post_matrix.sum(axis=0)

    max_par , err_hi , err_lo = estimate_confidence_interval(grid_kappa0 , prob_post_kappa0)
    log.info('max %5.5f +%5.5f -%5.5f', max_par, err_hi , err_lo)

    pl.figure()
    pl.plot(grid_kappa0 , prob_post_kappa0)
    pl.axvline(x=max_par - err_lo,linewidth=1, color='r')
    pl.axvline(x=max_par + err_hi,linewidth=1, color='r')
    pl.title('prob_post_kappa0')
    pl.figure()
    pl.plot(grid_radius , prob_post_radius)
    pl.title('prob_post_radius')



    import pdb; pdb.set_trace()

    # fitobj.plot_residual_whisker(best_model_g1, best_model_g2)
    # pl.suptitle('model post=% 10.4e kappa0=%5.2e radius=%2.4f' % (vmax_post,vmax_kappa0,vmax_radius) )
    # fitobj.plot_residual_g1g2(best_model_g1, best_model_g2)
    # pl.suptitle('model post=% 10.4e kappa0=%5.2e radius=%2.4f' % (vmax_post,vmax_kappa0,vmax_radius) )


    

    log.info('running mcmc')
    import pdb; pdb.set_trace()
    fitobj.n_samples=5000
    fitobj.run_mcmc()
    samples = fitobj.sampler.flatchain
    

    import pdb; pdb.set_trace()
    pl.figure()
    plotstools.plot_dist(samples)
    pl.show()

    pdf, par, _ = pl.hist(samples[:,0],bins=1000)
    estimate_confidence_interval(pdf, par)
    
    # pl.hist(fitobj.sampler.flatchain, bins=np.linspace(13,16,100), histtype="step")

def get_post_from_log(log_post):

    log_post = log_post - max(log_post.flatten())
    post = np.exp(log_post)
    norm = np.sum(post)
    post = post / norm

    return post


def main():

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    # parser.add_argument('-c', '--filename_config', default='test2.yaml',type=str, action='store', help='name of the yaml config file')
    # parser.add_argument('-d', '--dry', default=False,  action='store_true', help='Dry run, dont generate data')

    args = parser.parse_args()
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    log.setLevel(logging_level)

    # filaments_model_1f.log = log
    plotstools.log = log

    # grid_search(pair_info,shears_info)
    # test_model(shears_info,pair_info)

    # fit_single_halo()
    fit_single_filament()


main()