import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools, plotstools
import numpy as np
import pylab as pl
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import filaments_model_1h

log = logging.getLogger("filam..fit") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False


cospars = cosmology.cosmoparams()



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

    import pdb; pdb.set_trace()

    log_post , grid_M200, best_g1, best_g2, best_limit_mask = fitobj.run_gridsearch(M200_min=13.5,M200_max=15,M200_n=1000)
    log_post = log_post - max(log_post)
    norm = np.sum(np.exp(log_post))
    prob_post = np.exp(log_post) 
    pl.figure()
    pl.plot(grid_M200 , log_post , '.-')
    pl.figure()
    pl.plot(grid_M200 , prob_post , '.-')
    plotstools.adjust_limits()
    fitobj.plot_residual(best_g1, best_g2)
    pl.show()

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

    # pl.plot(fitobj.sampler.flatchain,'x')
    pl.show()

    median_m = [np.median(fitobj.sampler.flatchain)]
    print median_m
    fitobj.plot_model(median_m)


    import pdb;pdb.set_trace()

    # grid_search(pair_info,shears_info)
    # test_model(shears_info,pair_info)


main()