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

def add_nfw_to_random_points():
    
    import filaments_model_2hf, filament, nfw

    halo1 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos1.fits'))
    halo2 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos2.fits'))
    pairs_table = tabletools.loadTable(config['filename_pairs'])
    filename_shears_nfw = config['filename_shears'].replace('.pp2','.nfw.pp2')

    if os.path.isfile(filename_shears_nfw): 
        os.remove(filename_shears_nfw)
        logger.warning('overwriting file %s' , filename_shears_nfw)

    fitobj = filaments_model_2hf.modelfit()
    fitobj.get_bcc_pz(config['filename_pz'])
    prob_z = fitobj.prob_z
    grid_z_centers = fitobj.grid_z_centers
    grid_z_edges = fitobj.grid_z_edges

    for id_pair in range(len(pairs_table)):

        id_shear = pairs_table[id_pair]['ipair']
        logger.info('--------- pair %d shear %d --------' , id_pair, id_shear) 
        # now we use that
        id_pair_in_catalog = id_pair
        shears_info = tabletools.loadPickle(config['filename_shears'],pos=id_shear)
    
        fitobj = filaments_model_2hf.modelfit()
        fitobj.kappa_is_K = config['kappa_is_K']
        fitobj.prob_z = prob_z
        fitobj.grid_z_centers = grid_z_centers
        fitobj.grid_z_edges = grid_z_edges
        fitobj.shear_u_arcmin =  shears_info['u_arcmin']
        fitobj.shear_v_arcmin =  shears_info['v_arcmin']
        fitobj.shear_u_mpc =  shears_info['u_mpc']
        fitobj.shear_v_mpc =  shears_info['v_mpc']
        fitobj.shear_g1 =  shears_info['g1']
        fitobj.shear_g2 =  shears_info['g2']
        fitobj.shear_w =  shears_info['weight']
        fitobj.Dlos = pairs_table[id_pair]['Dlos']        
        fitobj.Dtot = np.sqrt(pairs_table[id_pair]['Dxy']**2+pairs_table[id_pair]['Dlos']**2)
        fitobj.boost = fitobj.Dtot/pairs_table[id_pair]['Dxy']
        fitobj.use_boost = config['use_boost']
        fitobj.R_start = config['R_start']

        # choose a method to add and account for noise
        if config['sigma_method'] == 'add':
            sigma_g_add =  config['sigma_add']
            fitobj.shear_g1 =  shears_info['g1'] + np.random.randn(len(shears_info['g1']))*sigma_g_add
            fitobj.shear_g2 =  shears_info['g2'] + np.random.randn(len(shears_info['g2']))*sigma_g_add
            fitobj.sigma_g =  np.std(fitobj.shear_g2,ddof=1)
            fitobj.sigma_ell = fitobj.sigma_g
            fitobj.inv_sq_sigma_g = 1./fitobj.sigma_g**2
            logger.info('added noise with level %f , using sigma_g=%2.5f' , sigma_g_add, fitobj.sigma_g)
        elif config['sigma_method'] == 'orig':
            fitobj.shear_n_gals = shears_info['n_gals']
            fitobj.inv_sq_sigma_g = fitobj.shear_w
            logger.info('using different sigma_g per pixel mean(inv_sq_sigma_g)=%2.5f len(inv_sq_sigma_g)=%d' , np.mean(fitobj.inv_sq_sigma_g) , len(fitobj.inv_sq_sigma_g))
        elif type(config['sigma_method'])==float:
            fitobj.shear_n_gals = shears_info['n_gals']
            fitobj.inv_sq_sigma_g =  shears_info['weight']**2 / ( shears_info['weight_sq'] * config['sigma_method']**2  )
            # remove infs
            fitobj.inv_sq_sigma_g[shears_info['weight_sq']<1e-8]=0
            logger.info('using constant sigma_g per pixel: sigma_e=%2.5f, mean(sigma_gp)=%2.5f  n_zeros=%d len(inv_sq_sigma_g)=%d n_nan=%d n_inf=%d' , config['sigma_method'], len(np.nonzero(shears_info['weight_sq']<1e-8)[0]),  np.mean(fitobj.inv_sq_sigma_g) , len(fitobj.inv_sq_sigma_g) , len( np.nonzero(np.isnan(fitobj.inv_sq_sigma_g))[0] ) , len( np.nonzero(np.isinf(fitobj.inv_sq_sigma_g))[0] ) )

                    
        fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair_in_catalog]
        fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair_in_catalog]
        fitobj.halo1_u_mpc =  pairs_table['u1_mpc'][id_pair_in_catalog]
        fitobj.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair_in_catalog]
        fitobj.halo1_z =  pairs_table['z'][id_pair_in_catalog]

        fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair_in_catalog]
        fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair_in_catalog]
        fitobj.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair_in_catalog]
        fitobj.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair_in_catalog]
        fitobj.halo2_z =  pairs_table['z'][id_pair_in_catalog]

        fitobj.n_model_evals = 0

        fitobj.pair_z  = (fitobj.halo1_z + fitobj.halo2_z) / 2.

        fitobj.filam = filament.filament()
        fitobj.filam.pair_z =fitobj.pair_z
        fitobj.filam.grid_z_centers = fitobj.grid_z_centers
        fitobj.filam.prob_z = fitobj.prob_z
        fitobj.filam.set_mean_inv_sigma_crit(fitobj.filam.grid_z_centers,fitobj.filam.prob_z,fitobj.filam.pair_z)

        fitobj.nh1 = nfw.NfwHalo()
        fitobj.nh1.z_cluster= fitobj.halo1_z
        fitobj.nh1.theta_cx = fitobj.halo1_u_arcmin
        fitobj.nh1.theta_cy = fitobj.halo1_v_arcmin 
        fitobj.nh1.set_mean_inv_sigma_crit(fitobj.grid_z_centers,fitobj.prob_z,fitobj.pair_z)

        fitobj.nh2 = nfw.NfwHalo()
        fitobj.nh2.z_cluster= fitobj.halo2_z
        fitobj.nh2.theta_cx = fitobj.halo2_u_arcmin
        fitobj.nh2.theta_cy = fitobj.halo2_v_arcmin 
        fitobj.nh2.set_mean_inv_sigma_crit(fitobj.grid_z_centers,fitobj.prob_z,fitobj.pair_z)


        mock_m200_h1 = pairs_table['m200_h1_fit'][id_pair_in_catalog]/1e14
        mock_m200_h2 = pairs_table['m200_h2_fit'][id_pair_in_catalog]/1e14

        mock_kappa0 = 0
        mock_radius = 2
        shear_model_g1, shear_model_g2, limit_mask , _ , _  = fitobj.draw_model([mock_kappa0, mock_radius, mock_m200_h1, mock_m200_h2])

        # pl.scatter(fitobj.shear_u_mpc,fitobj.shear_v_mpc,c=shear_model_g2); pl.colorbar(); pl.show()

        shears_info['g1'] = shears_info['g1'] + shear_model_g1
        shears_info['g2'] = shears_info['g2'] + shear_model_g2

        tabletools.savePickle(filename_shears_nfw,shears_info,append=True)

        logger.info('noise mean_g1=%2.2f mean_g2=%2.2f std_g1=%2.2f std_g2=%2.2f',np.mean(shears_info['g1']),np.mean(shears_info['g2']),np.std(shears_info['g1'],ddof=1), np.std(shears_info['g2'],ddof=1))







def main():
    
    available_actions = ['get_pairs','get_random_pairs','get_stamps','add_nfw']

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

    filename_halos  = config['filename_halos']
    filename_pairs  = config['filename_pairs']
    filename_shears = config['filename_shears']

    filaments_tools.add_phys_dist()
    if args.action == 'get_pairs':
        filaments_tools.get_pairs_topo()
        filaments_tools.stats_pairs()
    elif args.action == 'get_random_pairs':
        filaments_tools.get_pairs_resampling()
        filaments_tools.stats_pairs()
    elif args.action == 'get_stamps':
        filaments_tools.boundary_mpc=config['boundary_mpc']
        filaments_tools.get_shears_for_pairs(function_shears_for_single_pair=get_shears_for_single_pair,id_first=args.first,num=args.num)
    elif args.action == 'add_nfw':
        add_nfw_to_random_points()


    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

if __name__ == '__main__':

    main()
