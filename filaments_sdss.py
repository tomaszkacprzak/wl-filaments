import pyfits
import numpy as np
import pylab as pl
import scipy.interpolate as interp
import cosmology
import tabletools
import yaml, argparse, sys, logging 
from sklearn.neighbors import BallTree as BallTree
import galsim
import filaments_tools

logging.basicConfig(level=logging.INFO,format='%(message)s')
logger = logging.getLogger("filaments_sdss") 
logger.setLevel(logging.INFO)

dtype_pairs = { 'names' : ['ipair','ih1','ih2','DM','Dlos','Dxy'] , 'formats' : ['i8']*3 + ['f8']*3 }

dirname_data = '/home/kacprzak/data/'
filename_halos = 'sdss_halos.fits'
filename_pairs = 'sdss_pairs.fits'


def select_halos():

    halocat = tabletools.loadTable('DR7-Full.fits')

    # for now just save all LRGs
    tabletools.saveTable(filename_halos, halocat)



def get_pairs():

    (pairs_table, halos1, halos2) = filaments_tools.get_pairs(Dxy=[6,18],Dlos=6,filename_halos=filename_halos)

    tabletools.saveTable(filename_pairs,pairs_table)   
    tabletools.saveTable(filename_pairs.replace('fits','pairs_halos1_sdss.fits'), halos1)    
    tabletools.saveTable(filename_pairs.replace('fits','pairs_halos2_sdss.fits'), halos2)    


def main():

    description = 'filaments_sdss'
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
    logger = logging.getLogger("filaments_sdss") 
    logger.setLevel(logging_level)

    select_halos()
    filaments_tools.add_phys_dist(filename_halos=filename_halos)
    get_pairs()

main()