import os
if os.environ['USER'] == 'ucabtok': os.environ['MPLCONFIGDIR']='.'
import matplotlib as mpl 
if 'DISPLAY' not in os.environ:
    mpl.use('agg')
print 'using backend %s' % mpl.get_backend()
import yaml, argparse, sys, logging , time, pyfits , cosmology , tabletools
import numpy as np
import pylab as pl
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import filaments_tools

# logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
# logger = logging.getLogger("filaments_bcc") 

logger = logging.getLogger("fil..cfhtl") 
logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False




# dtype_shearbase = { 'names' : ['id','n_gals', 'file', 'min_ra','max_ra','min_dec','max_dec'] , 'formats' : ['i8']*2 + ['a1024']*1 + ['f8']*4 }
dtype_shearbase = { 'names' : ['id','n_gals', 'file','x','y','z'] , 'formats' : ['i8']*2 + ['a1024']*1 + ['f8']*3 }
dtype_shears_catalog = { 'names' : ['field' , 'ra' , 'dec' , 'e1' , 'e2' , 'weight' , 'ISOAREA_WORLD' , 'z' , 'star_flag' , 'MAG_r' ] , 
                            'formats' : [ 'a10' ] + ['f4']*7 + ['i4'] + ['f4'] }

dirname_data = os.getenv("HOME") + '/data/'
filename_shearbase = 'shear_base.txt'

shear1_col = 's1'
shear2_col = 's2'

cfhtlens_shear_catalog = None


def get_shears_for_single_pair(halo1,halo2,idp=0):

    global cfhtlens_shear_catalog
    if cfhtlens_shear_catalog == None:
        filename_cfhtlens_shears =  config['filename_cfhtlens_shears']

        cfhtlens_shear_catalog = tabletools.loadTable(filename_cfhtlens_shears)
        if 'star_flag' in cfhtlens_shear_catalog.dtype.names:
            select = cfhtlens_shear_catalog['star_flag'] == 0
            cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
            select = cfhtlens_shear_catalog['fitclass'] == 0
            cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
            logger.info('removed stars, remaining %d' , len(cfhtlens_shear_catalog))

            select = (cfhtlens_shear_catalog['e1'] != 0.0) * (cfhtlens_shear_catalog['e2'] != 0.0)
            cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
            logger.info('removed zeroed shapes, remaining %d' , len(cfhtlens_shear_catalog))

    # correcting additive systematics
    if 'e1corr' in cfhtlens_shear_catalog.dtype.names:       
        shear_g1 , shear_g2 = cfhtlens_shear_catalog['e1corr'] , -cfhtlens_shear_catalog['e2corr']
        shear_ra_deg , shear_de_deg , shear_z = cfhtlens_shear_catalog['ALPHA_J2000'] , cfhtlens_shear_catalog['DELTA_J2000'] ,  cfhtlens_shear_catalog['Z_B']
    else:
        shear_g1 , shear_g2 = cfhtlens_shear_catalog['e1'] , -(cfhtlens_shear_catalog['e2']  - cfhtlens_shear_catalog['c2'])
        shear_ra_deg , shear_de_deg , shear_z = cfhtlens_shear_catalog['ra'] , cfhtlens_shear_catalog['dec'] ,  cfhtlens_shear_catalog['z']

    halo1_ra_deg , halo1_de_deg = halo1['ra'],halo1['dec']
    halo2_ra_deg , halo2_de_deg = halo2['ra'],halo2['dec']

    pair_ra_deg,  pair_de_deg = cosmology.get_midpoint(halo1_ra_deg , halo1_de_deg , halo2_ra_deg , halo2_de_deg,unit='deg')
    pair_z = np.mean([halo1['z'],halo2['z']])

    pairs_shear , halos_coords , pairs_shear_full   = filaments_tools.create_filament_stamp(halo1_ra_deg, halo1_de_deg, 
                            halo2_ra_deg, halo2_de_deg, 
                            shear_ra_deg, shear_de_deg, 
                            shear_g1, shear_g2, shear_z, 
                            pair_z, lenscat=cfhtlens_shear_catalog , shear_bias_m=cfhtlens_shear_catalog['m'] , shear_weight=cfhtlens_shear_catalog['weight'] )

    if len(pairs_shear) < 100:
        logger.error('found only %d shears' % len(pairs_shear))
        return None , None , None

    return pairs_shear , halos_coords, pairs_shear_full

def get_pairs_null1(filename_pairs_null1 = 'pairs_cfhtlens_null1.fits', filename_pairs_exclude = 'pairs_cfhtlens.fits', filename_halos='halos_cfhtlens.fits',range_Dxy=[6,10]):

    pairs_table, halos1, halos2 = filaments_tools.get_pairs_null1(range_Dxy=range_Dxy,Dlos=6,filename_halos=filename_halos,filename_pairs_exclude=filename_pairs_exclude)

    tabletools.saveTable(filename_pairs_null1,pairs_table)   
    tabletools.saveTable(filename_pairs_null1.replace('.fits','.halos1.fits'), halos1)    
    tabletools.saveTable(filename_pairs_null1.replace('.fits','.halos2.fits'), halos2)    

    # import pylab as pl
    # pl.scatter(pairs_table['ra1'] , pairs_table['dec1'])
    # pl.scatter(pairs_table['ra2'] , pairs_table['dec2'])
    # pl.show()


def get_pairs_null1(filename_pairs_null1 = 'pairs_cfhtlens_null1.fits', filename_pairs = 'pairs_cfhtlens.fits', filename_halos='halos_cfhtlens.fits',range_Dxy=[6,10]):

    if config['pair_last'] == -1:
        n_unpaired = -1
    else:    
        n_unpaired =config['pair_last'] - config['pair_first']


    pairs_table, halos1, halos2 = filaments_tools.get_pairs_null1(range_Dxy=range_Dxy,Dlos=6,filename_halos=filename_halos,n_unpaired=n_unpaired)

    tabletools.saveTable(filename_pairs_null1,pairs_table)   
    tabletools.saveTable(filename_pairs_null1.replace('.fits','.halos1.fits'), halos1)    
    tabletools.saveTable(filename_pairs_null1.replace('.fits','.halos2.fits'), halos2)    

    n_pairs = len(pairs_table)
    return n_pairs


    # import pylab as pl
    # pl.scatter(pairs_table['ra1'] , pairs_table['dec1'])
    # pl.scatter(pairs_table['ra2'] , pairs_table['dec2'])
    # pl.show()

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
    
    available_actions = ['main','random']

    global config , args

    description = 'filaments_cfhtlens'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    parser.add_argument('-c', '--filename_config', default='cfhtlens.yaml',type=str, action='store', help='name of the yaml config file')
    parser.add_argument('-f', '--first', default=0,type=int, action='store', help='first pairs to process')
    parser.add_argument('-n', '--num', default=-1 ,type=int, action='store', help='last pairs to process')
    parser.add_argument('-a', '--action', default='main' ,type=str, action='store', help='what to do? %s' % available_actions)
    # parser.add_argument('-d', '--dry', default=False,  action='store_true', help='Dry run, dont generate data')

    args = parser.parse_args()
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    logger.setLevel(logging_level)
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


    config = yaml.load(open(args.filename_config))
    filaments_tools.config = config

    filename_halos=config['filename_halos']
    filename_pairs = config['filename_pairs']
    filename_shears = config['filename_shears']

    filaments_tools.add_phys_dist()
    if args.action == 'main':
        filaments_tools.get_pairs_topo()
    elif args.action == 'random':
        filaments_tools.get_pairs_resampling()

    filaments_tools.stats_pairs()
    filaments_tools.boundary_mpc=config['boundary_mpc']
    filaments_tools.get_shears_for_pairs(function_shears_for_single_pair=get_shears_for_single_pair,id_first=args.first,num=args.num)

    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

if __name__ == '__main__':

    main()
