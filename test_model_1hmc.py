import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools
import numpy as np
import pylab as pl
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import filaments_model_1hmc


logging.basicConfig(filename='filament_fit.log',level=logging.DEBUG,format='%(message)s')
log = logging.getLogger("get_halo_pairs") 
log.setLevel(logging.DEBUG)

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
    logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    log = logging.getLogger("filaments_fit") 
    log.setLevel(logging_level)

    id_pair = 7
    filename_shears = 'shears_bcc_g.%03d.fits' % id_pair
    filename_pairs = 'pairs_bcc.fits'
    filename_halo1 = 'pairs_bcc.halos1.fits'

    pairs_table = tabletools.loadTable(filename_pairs)
    shears_info = tabletools.loadTable(filename_shears)
    halo1_table = tabletools.loadTable(filename_halo1)


    fitobj = filaments_model_1hmc.modelfit()
    fitobj.sigma_g =  0.01
    fitobj.shear_g1 =  shears_info['g1sc'] + np.random.randn(len(shears_info['g1sc']))*fitobj.sigma_g
    fitobj.shear_g2 =  shears_info['g2sc'] + np.random.randn(len(shears_info['g1sc']))*fitobj.sigma_g
    fitobj.shear_u_arcmin =  shears_info['u_arcmin']
    fitobj.shear_v_arcmin =  shears_info['v_arcmin']
    fitobj.halo_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
    fitobj.halo_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
    fitobj.halo_z =  pairs_table['z'][id_pair]

    pair_info = pairs_table[id_pair]

    fitobj.run_mcmc()
    print fitobj.sampler
    print halo1_table['m200'][id_pair]

    pl.figure()
    pl.hist(fitobj.sampler.flatchain[:,0], 100, color="k", histtype="step")
    pl.figure
    pl.hist(fitobj.sampler.flatchain[:,1], 100, color="k", histtype="step")
    pl.show()


    import pdb; pdb.set_trace()

main()