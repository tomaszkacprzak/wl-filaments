import os
import matplotlib as mpl 
if 'DISPLAY' not in os.environ:
    mpl.use('agg')
print 'using backend %s' % mpl.get_backend()
import yaml, argparse, sys, logging , time, pyfits , cosmology , tabletools
import numpy as np
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import filaments_tools

# logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
# logger = logging.getLogger("filaments_bcc") 

logger = logging.getLogger("filam..bcc") 
logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False




# dtype_shearbase = { 'names' : ['id','n_gals', 'file', 'min_ra','max_ra','min_dec','max_dec'] , 'formats' : ['i8']*2 + ['a1024']*1 + ['f8']*4 }
dtype_shearbase = { 'names' : ['id','n_gals', 'file','x','y','z'] , 'formats' : ['i8']*2 + ['a1024']*1 + ['f8']*3 }

dirname_data = os.getenv("HOME") + '/data/'
filename_shearbase = 'shear_base.txt'

def fix_case(arr):

    arr.dtype.names = [n.lower() for n in arr.dtype.names]

def get_pairs_null1(filename_pairs_null1 = 'pairs_bcc_null1.fits', filename_pairs = 'pairs_bcc.fits', filename_halos='halos_bcc.fits',range_Dxy=[6,10]):

    n_unpaired =config['pair_last'] - config['pair_first']
    pairs_table, halos1, halos2 = filaments_tools.get_pairs_null1(range_Dxy=range_Dxy,Dlos=6,filename_halos=filename_halos,n_unpaired=n_unpaired)

    tabletools.saveTable(filename_pairs_null1,pairs_table)   
    tabletools.saveTable(filename_pairs_null1.replace('.fits','.halos1.fits'), halos1)    
    tabletools.saveTable(filename_pairs_null1.replace('.fits','.halos2.fits'), halos2)    

    # import pylab as pl
    # pl.scatter(pairs_table['ra1'] , pairs_table['dec1'])
    # pl.scatter(pairs_table['ra2'] , pairs_table['dec2'])
    # pl.show()


def get_pairs(filename_pairs = 'pairs_bcc.fits',filename_halos='halos_bcc.fits',range_Dxy=[6,10]):

    pairs_table, halos1, halos2 = filaments_tools.get_pairs(range_Dxy=range_Dxy,Dlos=6,filename_halos=filename_halos)

    tabletools.saveTable(filename_pairs,pairs_table)   
    tabletools.saveTable(filename_pairs.replace('.fits','.halos1.fits'), halos1)    
    tabletools.saveTable(filename_pairs.replace('.fits','.halos2.fits'), halos2)    


def select_halos(range_z=[0.1,0.6],range_M=[1e14,1e16],filename_halos='halos_bcc.fits',n_bcc_halo_files=10):

    logger.info('selecting halos in range_z (%2.2f,%2.2f) and range_M (%2.2e,%2.2e)' % (range_z[0],range_z[1],range_M[0],range_M[1]))

    files = np.loadtxt('filelist_halos.txt',dtype={'names': ['file'], 'formats': ['a100'] })

    big_catalog = []
    for filename in files['file'][:n_bcc_halo_files]:
        filename_aardvark_halos = '%s/BCC/bcc_a1.0b/aardvark_v1.0/halos/%s' % (dirname_data,filename)
        halocat = pyfits.getdata(filename_aardvark_halos)
        logger.info('%s total number of halos %d' % (filename,len(halocat)))

        # select on Z
        select = (halocat['Z'] > range_z[0]) * (halocat['Z'] < range_z[1])
        halocat=halocat[select]
        logger.info('selected on Z number of halos: %d' % len(halocat))

        # select on M
        select = (halocat['M200'] > range_M[0]) * (halocat['M200'] < range_M[1])
        halocat=halocat[select]
        logger.info('selected on M200 number of halos: %d' % len(halocat))

        big_catalog.append(np.array(halocat))

    big_catalog = np.concatenate(big_catalog)
    index=range(0,len(big_catalog))
    big_catalog = tabletools.appendColumn(big_catalog,'id',index,dtype='i8')
    fix_case( big_catalog )

    logger.info('number of halos %d', len(big_catalog))
    tabletools.saveTable(filename_halos,big_catalog)
    logger.info('wrote %s' % filename_halos)

def get_shear_files_catalog():


    filelist_shears = np.loadtxt('filelist_shears.txt',dtype='a1024')  

    list_shearbase = []
    total_n_gals = 0
    for ix,fs in enumerate(filelist_shears):
        logger.info('%3d\t%s\t%1.2e' % ( ix, fs ,float(total_n_gals) ))
        shear_cat_full=tabletools.loadTable(fs)
        # use every 100 shear for speed
        shear_cat = shear_cat_full[::100]
        radius = shear_cat['ra']*0 + 1 # set to 1
        # xyz = cosmology.get_euclidian_coords(shear_cat['ra'], shear_cat['dec'] , radius)
        xs,ys,zs = cosmology.spherical_to_cartesian_deg(shear_cat['ra'], shear_cat['dec'] , radius)  
        x,y,z = np.mean(xs), np.mean(ys) , np.mean(zs)
        del(xs) 
        del(ys)
        del(zs)
        row = np.array([(ix, len(shear_cat_full),fs,x,y,z )],dtype=dtype_shearbase)        
        total_n_gals += len(shear_cat_full)
        list_shearbase.append(row)
        del(shear_cat)
        del(shear_cat_full)

    logger.info('total gals %d',total_n_gals)
    shearbase = np.concatenate(list_shearbase)
    tabletools.saveTable(filename_shearbase,shearbase)


def get_shears_for_single_pair(halo1,halo2,idp=0):

        logger.debug('ra=(%2.2f,%2.2f) dec=(%2.2f,%2.2f) ' % (halo1['ra'],halo2['ra'],halo1['dec'],halo2['dec']))

        shear_base = tabletools.loadTable(filename_shearbase,dtype=dtype_shearbase)
       
        redshift_offset = 0.2
        # pair_dra = np.abs(halo1['ra'] - halo2['ra'])
        # pair_ddec = np.abs(halo1['dec'] - halo2['dec'])

        halo1_ra_deg , halo1_de_deg = halo1['ra'],halo1['dec']
        halo2_ra_deg , halo2_de_deg = halo2['ra'],halo2['dec']

        pair_ra_deg,  pair_de_deg = cosmology.get_midpoint_deg(halo1_ra_deg , halo1_de_deg , halo2_ra_deg , halo2_de_deg)
        pair_z = np.mean([halo1['z'],halo2['z']])
        
        # find the corresponding files

        radius = 1.
        # pair_xyz = cosmology.get_euclidian_coords( pair_ra_deg , pair_de_deg , radius)
        x,y,z = cosmology.spherical_to_cartesian_deg( pair_ra_deg , pair_de_deg , radius)
        pair_xyz = np.array([x , y , z])
        box_coords_x = shear_base['x']
        box_coords_y = shear_base['y']
        box_coords_z = shear_base['z']

        box_coords_xyz = np.concatenate([ box_coords_x[:,None], box_coords_y[:,None], box_coords_z[:,None] ] , axis=1)
        logger.info('getting Ball Tree for 3D')
        BT = BallTree(box_coords_xyz, leaf_size=5)
        n_connections=5
        bt_dx,bt_id = BT.query(pair_xyz,k=n_connections)

        list_set = []

        for iset, vset in enumerate( shear_base[bt_id]['file'][0] ):
            # vset=vset.replace('kacprzak','tomek')
            lenscat=tabletools.loadTable(vset)
            # prelim cut on z
            select =  lenscat['z'] > (pair_z + redshift_offset)
            lenscat=lenscat[select]

            list_set.append(lenscat)
            logger.debug('opened %s with %d gals mean_ra=%2.2f, mean_de=%2.2f' % (vset,len(lenscat),np.mean(lenscat['ra']),np.mean(lenscat['dec'])))

        shear1_col = config['shear1_col']
        shear2_col = config['shear2_col']

        lenscat_all = np.concatenate(list_set)
        shear_g1 , shear_g2 = -lenscat_all[shear1_col] , lenscat_all[shear2_col] 
        shear_ra_deg , shear_de_deg , shear_z = lenscat_all['ra'] , lenscat_all['dec'] ,  lenscat_all['z']

        # import pylab as pl
        # pl.figure()
        # pl.scatter(pair_ra_deg      , pair_de_deg  , 100, 'r' , marker='x')
        # pl.scatter(halo1_ra_deg     , halo1_de_deg , 100, 'c' , marker='o')
        # pl.scatter(halo2_ra_deg     , halo2_de_deg , 100, 'm' , marker='o')
        # select = np.random.permutation(len(shear_ra_deg))[:10000]
        # pl.scatter(shear_ra_deg[select]  , shear_de_deg[select] ,1,  'm' , marker='.')
        # pl.show()

        pairs_shear , halos_coords , pairs_shear_full   = filaments_tools.create_filament_stamp(halo1_ra_deg, halo1_de_deg, 
                                halo2_ra_deg, halo2_de_deg, 
                                shear_ra_deg, shear_de_deg, 
                                shear_g1, shear_g2, shear_z, 
                                pair_z, lenscat_all )

        if len(pairs_shear) < 100:
            logger.error('found only %d shears' % len(pairs_shear))
            return None , None

        return pairs_shear , halos_coords, pairs_shear_full
 


def main():

    description = 'filaments_bcc'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    parser.add_argument('-c', '--filename_config', default='filaments_config.yaml',type=str, action='store', help='name of the yaml config file')
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
    # n_pairs = config['n_pairs']
    filename_halos=config['filename_halos']
    filename_pairs = config['filename_pairs']
    filename_shears = config['filename_shears']
        
    # get_shear_files_catalog()
    
    if config['mode'] == 'pairs':

        select_halos(filename_halos=filename_halos,range_M=range_M,n_bcc_halo_files=config['n_bcc_halo_files'])
        filaments_tools.add_phys_dist(filename_halos=filename_halos)
        get_pairs(filename_halos=filename_halos, filename_pairs=filename_pairs, range_Dxy=range_Dxy)

    elif config['mode'] == 'null1_unpaired':

        filename_pairs_exclude = config['filename_pairs_exclude']
        get_pairs_null1(filename_pairs_null1 = filename_pairs, filename_pairs = filename_pairs_exclude ,  filename_halos=filename_halos , range_Dxy=range_Dxy)

    filaments_tools.stats_pairs(filename_pairs=filename_pairs)
    filaments_tools.boundary_mpc=config['boundary_mpc']
    
    logger.info('getting shear catalogs with shear columns %s %s' % (config['shear1_col'],config['shear2_col']))
    filaments_tools.get_shears_for_pairs(filename_pairs=filename_pairs, filename_shears=filename_shears, function_shears_for_single_pair=get_shears_for_single_pair,id_first=config['pair_first'],id_last=config['pair_last'])
    # filaments_tools.get_halo_map(filename_pairs_bcc_null1)

    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

main()