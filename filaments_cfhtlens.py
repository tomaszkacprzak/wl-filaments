import os
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

logger = logging.getLogger("fil..cfhtlens") 
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
        cfhtlens_shear_catalog = tabletools.loadTable(os.environ['HOME']+ '/data/CFHTLens/cfhtlens_cat_full.fits')
        select = cfhtlens_shear_catalog['star_flag'] == 0
        cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
        logger.info('removed stars, remaining %d' , len(cfhtlens_shear_catalog))

        select = (cfhtlens_shear_catalog['e1'] != 0.0) * (cfhtlens_shear_catalog['e2'] != 0.0)
        cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
        logger.info('removed zeroed shapes, remaining %d' , len(cfhtlens_shear_catalog))

    shear_g1 , shear_g2 = cfhtlens_shear_catalog['e1'] , cfhtlens_shear_catalog['e2'] 
    shear_ra_deg , shear_de_deg , shear_z = cfhtlens_shear_catalog['ra'] , cfhtlens_shear_catalog['dec'] ,  cfhtlens_shear_catalog['z']

    halo1_ra_deg , halo1_de_deg = halo1['ra'],halo1['dec']
    halo2_ra_deg , halo2_de_deg = halo2['ra'],halo2['dec']

    pair_ra_deg,  pair_de_deg = cosmology.get_midpoint_deg(halo1_ra_deg , halo1_de_deg , halo2_ra_deg , halo2_de_deg)
    pair_z = np.mean([halo1['z'],halo2['z']])


    pairs_shear , halos_coords , pairs_shear_full   = filaments_tools.create_filament_stamp(halo1_ra_deg, halo1_de_deg, 
                            halo2_ra_deg, halo2_de_deg, 
                            shear_ra_deg, shear_de_deg, 
                            shear_g1, shear_g2, shear_z, 
                            pair_z, cfhtlens_shear_catalog )

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

    # import pylab as pl
    # pl.scatter(pairs_table['ra1'] , pairs_table['dec1'])
    # pl.scatter(pairs_table['ra2'] , pairs_table['dec2'])
    # pl.show()

def get_pairs(filename_pairs = 'pairs_cfhtlens.fits',filename_halos='halos_cfhtlens.fits',range_Dxy=[6,10]):

    pairs_table, halos1, halos2 = filaments_tools.get_pairs(range_Dxy=range_Dxy,Dlos=6,filename_halos=filename_halos)

    tabletools.saveTable(filename_pairs,pairs_table)   
    tabletools.saveTable(filename_pairs.replace('.fits','.halos1.fits'), halos1)    
    tabletools.saveTable(filename_pairs.replace('.fits','.halos2.fits'), halos2)    

def fix_case(arr):

    arr.dtype.names = [n.lower() for n in arr.dtype.names]


def select_halos(range_z=[0.1,0.6],range_M=[2,10],filename_halos='halos_cfhtlens.fits'):

    logger.info('selecting halos in range_z (%2.2f,%2.2f) and range_M (%2.2e,%2.2e)' % (range_z[0],range_z[1],range_M[0],range_M[1]))

    filename_halos_cfhtlens = os.environ['HOME'] + '/data/CFHTLens/CFHTLS_wide_clusters_Durret2011/wide.fits'
    halocat = tabletools.loadTable(filename_halos_cfhtlens)

    # filename_halos_bcc = 'halos_bcc.fits'
    # halos_bcc = pyfits.getdata(filename_halos_bcc)

    # bins_snr = np.array([0,1,2,3,4,5,6,7])+0.5
    # pl.hist(halos_cfhtlens['snr'],histtype='step',bins=bins_snr)
    # pl.hist(np.log10(halos_bcc['m200']),histtype='step')
    # pl.show()


    # # select on Z
    select = (halocat['zphot'] > range_z[0]) * (halocat['zphot'] < range_z[1])
    halocat=halocat[select]
    logger.info('selected on Z number of halos: %d' % len(halocat))

    # # select on M
    select = (halocat['snr'] > range_M[0]) * (halocat['snr'] < range_M[1])
    halocat=halocat[select]
    logger.info('selected on SNR number of halos: %d' % len(halocat))

    index=range(0,len(halocat))
    halocat = tabletools.appendColumn(halocat,'id',index,dtype='i8')
    fix_case( halocat )
    halocat.dtype.names = ('field', 'index', 'ra', 'dec', 'z', 'snr', 'id')

    logger.info('number of halos %d', len(halocat))
    tabletools.saveTable(filename_halos,halocat)
    logger.info('wrote %s' % filename_halos)

    pl.subplot(1,3,1)
    bins_snr = np.array([0,1,2,3,4,5,6,7])+0.5
    pl.hist(halocat['snr'],histtype='step',bins=bins_snr)

    pl.subplot(1,3,2)
    pl.hist(halocat['z'],histtype='step',bins=200)

    pl.subplot(1,3,3)
    pl.scatter(halocat['z'],halocat['snr'])

    filename_fig = 'figs/cfhtlens.hists.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' , filename_fig)


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

    description = 'filaments_cfhtlens'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    parser.add_argument('-c', '--filename_config', default='cfhtlens.yaml',type=str, action='store', help='name of the yaml config file')
    parser.add_argument('-f', '--first', default=0,type=int, action='store', help='first pairs to process')
    parser.add_argument('-l', '--last', default=-1 ,type=int, action='store', help='last pairs to process')
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


    global config 
    config = yaml.load(open(args.filename_config))
    filaments_tools.config = config

    range_M = map(float,config['range_M'])
    range_Dxy = map(float,config['range_Dxy'])
    range_z = map(float,config['range_z'])
    filename_halos=config['filename_halos']
    filename_pairs = config['filename_pairs']
    filename_shears = config['filename_shears']

    if config['mode'] == 'pairs':

        select_halos(filename_halos=filename_halos,range_M=range_M,range_z=range_z)
        filaments_tools.add_phys_dist(filename_halos=filename_halos)
        get_pairs(filename_halos=filename_halos, filename_pairs=filename_pairs, range_Dxy=range_Dxy)

    elif (config['mode'] == 'null1_unpaired') or (config['mode'] == 'null1_all'):

        filename_pairs_exclude = config['filename_pairs_exclude']
        get_pairs_null1(filename_pairs_null1 = filename_pairs, filename_pairs_exclude = filename_pairs_exclude ,  filename_halos=filename_halos , range_Dxy=range_Dxy)

    filaments_tools.stats_pairs(filename_pairs=filename_pairs)
    filaments_tools.boundary_mpc=config['boundary_mpc']

    logger.info('getting noisy shear catalogs')
    filaments_tools.get_shears_for_pairs(filename_pairs=filename_pairs, filename_shears=filename_shears, function_shears_for_single_pair=get_shears_for_single_pair,id_first=config['pair_first'],id_last=config['pair_last'])

    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

main()