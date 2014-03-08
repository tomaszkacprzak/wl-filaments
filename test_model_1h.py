import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools, plotstools
import numpy as np
import pylab as pl
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import filaments_model_1h

cospars = cosmology.cosmoparams()


def main():

    description = 'test_mod1h'
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
    logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    log = logging.getLogger("test_mod1h") 
    log.setLevel(logging_level)

    id_pair = 7
    filename_shears = 'shears_bcc_g.%03d.fits' % id_pair
    filename_pairs = 'pairs_bcc.fits'
    filename_halo1 = 'pairs_bcc.halos1.fits'

    pairs_table = tabletools.loadTable(filename_pairs)
    shears_info = tabletools.loadTable(filename_shears)
    halo1_table = tabletools.loadTable(filename_halo1)

    true_M200 = 15


    fitobj = filaments_model_1h.modelfit()
    # filaments_model_1h.log.setLevel(logging.DEBUG)
    fitobj.shear_z = 1
    fitobj.shear_u_arcmin =  shears_info['u_arcmin']
    fitobj.shear_v_arcmin =  shears_info['v_arcmin']
    fitobj.halo_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
    fitobj.halo_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
    fitobj.halo_z =  pairs_table['z'][id_pair]
    fitobj.sigma_g =  0.1
    fitobj.shear_g1 , fitobj.shear_g2 , limit_mask =  fitobj.draw_model([true_M200])
    fitobj.shear_g1 = fitobj.shear_g1 + np.random.randn(len(fitobj.shear_g1))*fitobj.sigma_g
    fitobj.shear_g2 = fitobj.shear_g2 + np.random.randn(len(fitobj.shear_g2))*fitobj.sigma_g
    fitobj.plot_model([true_M200])
    filename_fig = 'halo_model_true.png'
    pl.savefig(filename_fig)
    log.info('saved %s' % filename_fig)
    pl.show()

    pair_info = pairs_table[id_pair]

    log_post , grid_M200 = fitobj.run_gridsearch()
    log_post = log_post - max(log_post)
    norm = np.sum(np.exp(log_post))
    prob_post = np.exp(log_post) 
    pl.figure()
    pl.plot(grid_M200 , log_post , '.-')
    pl.figure()
    pl.plot(grid_M200 , prob_post , '.-')
    plotstools.adjust_limits()
    pl.show()

    fitobj.run_mcmc()
    pl.figure()
    pl.hist(fitobj.sampler.flatchain, bins=np.linspace(13,18,100), color="k", histtype="step")
    # pl.plot(fitobj.sampler.flatchain,'x')
    pl.show()
    median_m = [np.median(fitobj.sampler.flatchain)]
    print median_m
    fitobj.plot_model(median_m)
    filename_fig = 'halo_model_median.png'
    pl.savefig(filename_fig)
    log.info('saved %s' % filename_fig)


    import pdb;pdb.set_trace()

    # grid_search(pair_info,shears_info)
    # test_model(shears_info,pair_info)


main()