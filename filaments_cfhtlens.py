import pyfits
import numpy as np
import pylab as pl
import scipy.interpolate as interp
import cosmology
import tabletools
import yaml, argparse, sys, logging 
from sklearn.neighbors import BallTree as BallTree
import galsim

logging.basicConfig(level=logging.INFO,format='%(message)s')
logger = logging.getLogger("get_halo_pairs") 
logger.setLevel(logging.INFO)

dtype_pairs = { 'names' : ['ipair','ih1','ih2','DM','Dlos','Dxy'] , 'formats' : ['i8']*3 + ['f8']*3 }


def estimate_snr():

    pairs_table = hp.get_pairs(Dxy=[6,18],Mstar=[3e13,1e16],zbin=[0.01,0.9],filename_halos='wide.fits',n_sigma=3)
    n_pairs=len(pairs_table)
    print 'n_pairs', n_pairs
    n_pairs_sdss = 200000
    n_eff_sdss = 0.5
    n_eff_cfht = 15
    n_sigma_sdss = 10
    kernel_gain_cfht = 3


    n_eff_pairs_sdss = n_pairs_sdss*n_eff_sdss
    print 'n_eff_pairs_sdss', n_eff_pairs_sdss
    n_eff_pairs_cfht = n_pairs*n_eff_cfht*kernel_gain_cfht
    print 'n_eff_pairs_cfht', n_eff_pairs_cfht

    sigma_single_filament = np.sqrt(n_eff_pairs_sdss)/n_sigma_sdss
    n_sigma_cfht = np.sqrt(n_eff_pairs_cfht)/sigma_single_filament

    print 'n_sigma_cfht', n_sigma_cfht

    filename_durret = 'wide.fits'
    durret_clusters = tabletools.loadTable(filename_durret)
    pl.hist(durret_clusters['snr'],bins=range(1,10))
    pl.show()



def main():

    

    global logger , config , args

    description = 'filaments_bcc'
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
    logger = logging.getLogger("filaments_bcc") 
    logger.setLevel(logging_level)