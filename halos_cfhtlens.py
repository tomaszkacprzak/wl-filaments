import os
if os.environ['USER'] == 'ucabtok': os.environ['MPLCONFIGDIR']='.'
import matplotlib as mpl
if 'DISPLAY' not in os.environ:
    mpl.use('agg')
import os, yaml, argparse, sys, logging , pyfits, tabletools, cosmology, filaments_tools, plotstools, mathstools, scipy, scipy.stats, time
import numpy as np
import pylab as pl
print 'using matplotlib backend' , pl.get_backend()
# import matplotlib as mpl;
# from matplotlib import figure;
pl.rcParams['image.interpolation'] = 'nearest' ; 
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import cPickle as pickle
import filaments_model_1h
import filaments_model_1f
import filaments_model_2hf
import shutil

logger = logging.getLogger("filam..fit") 
logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False

cospars = cosmology.cosmoparams()

dtype_halostamp = {'names':['n_gals','u_arcmin','v_arcmin','g1','g2','weight','hist_g1','hist_g2' ,'hist_m'],'formats': ['i4']+['f4']*8}


def fix_case(arr):

    arr.dtype.names = [n.lower() for n in arr.dtype.names]

def select_halos():

    range_z=config['range_z']
    range_M=config['range_M']
    filename_halos=config['filename_halos']


    logger.info('selecting halos in range_z (%2.2f,%2.2f) and range_M (%2.2e,%2.2e)' % (range_z[0],range_z[1],range_M[0],range_M[1]))

    filename_halos_cfhtlens = os.environ['HOME'] + '/data/CFHTLens/CFHTLS_wide_clusters_Durret2011/wide.fits'
    halocat = tabletools.loadTable(filename_halos_cfhtlens)

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

    filename_fig = 'figs/hist.cfhtlens.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' , filename_fig)


def select_halos_LRGSCLUS():

    range_z=config['range_z']
    range_M=config['range_M']
    filename_halos=config['filename_halos']


    import os
    import numpy as np
    import pylab as pl
    import tabletools
    import pyfits
    import plotstools

    filename_catalog_lrgs = 'lrgs_cfhtlens_lrg.fits'
    filename_catalog_clusters = os.environ['HOME'] + '/data/CFHTLens/ClusterZ/clustersz.fits'

    cat_clusters=tabletools.loadTable(filename_catalog_clusters)
    cat_lrgs=np.array(pyfits.getdata(filename_catalog_lrgs))

    select= (cat_clusters['m200']>range_M[0]) * (cat_clusters['m200']<range_M[1])
    cat_clusters=cat_clusters[select]

    coords_lrgs = np.concatenate([cat_lrgs['ra'][:,None],cat_lrgs['dec'][:,None]],axis=1)
    coords_clusters = np.concatenate([cat_clusters['ra'][:,None],cat_clusters['dec'][:,None]],axis=1)
    from sklearn.neighbors import BallTree as BallTree
    BT = BallTree(coords_lrgs, leaf_size=5)
    n_connections=1
    bt_dx,bt_id = BT.query(coords_clusters,k=n_connections)
    
    dx = 0.02
    select=bt_dx<dx
    ids_selected = bt_id[select]

    cat_lrgs_lrgsclus = cat_lrgs[ids_selected]
    cat_clus_lrgsclus = cat_clusters[select.flatten()]

    select = np.abs(cat_lrgs_lrgsclus['z'] - cat_clus_lrgsclus['z'])<0.1
    cat_lrgs_lrgsclus = cat_lrgs_lrgsclus[select]
    cat_clus_lrgsclus = cat_clus_lrgsclus[select]

    cat_join=cat_clus_lrgsclus.copy()
    cat_join['z'] = cat_lrgs_lrgsclus['z']
    print 'found %d lrg-clus matches' % len(cat_join)

    select= (cat_join['z'] > range_z[0]) * (cat_join['z'] < range_z[1])
    cat_join = cat_join[select]
    fix_case(cat_join)
    print 'selected on redshift range, n_clusters=%d' % len(cat_join)

    pl.figure()
    pl.hist(cat_join['m200'])
    filename_fig = 'figs/hist.m200.png'
    pl.savefig(filename_fig)
    pl.close()
    print 'saved' , filename_fig

    tabletools.saveTable(filename_halos,cat_join)

def select_halos_CLUSTERZ():

    range_z=config['range_z']
    range_M=config['range_M']
    filename_halos=config['filename_halos']


    logger.info('selecting halos in range_z (%2.2f,%2.2f) and range_M (%.2f,%.2f)' % (range_z[0],range_z[1],range_M[0],range_M[1]))

    filename_halos_cfhtlens = os.environ['HOME'] + '/data/CFHTLens/ClusterZ/clustersz.fits'
    halocat = tabletools.loadTable(filename_halos_cfhtlens)

    # # select on Z
    select = (halocat['z'] > range_z[0]) * (halocat['z'] < range_z[1])
    halocat=halocat[select]
    logger.info('selected on Z number of halos: %d' % len(halocat))

    # # select on M
    select = (halocat['m200'] > range_M[0]) * (halocat['m200'] < range_M[1])
    halocat=halocat[select]
    logger.info('selected on m200 number of halos: %d' % len(halocat))

    fix_case( halocat )
   
    logger.info('number of halos %d', len(halocat))
    tabletools.saveTable(filename_halos,halocat)
    logger.info('wrote %s' % filename_halos)

    pl.subplot(1,3,1)
    bins_snr = np.linspace(13,16,20)
    pl.hist(halocat['m200'],histtype='step',bins=bins_snr)

    pl.subplot(1,3,2)
    pl.hist(halocat['z'],histtype='step',bins=200)

    pl.subplot(1,3,3)
    pl.scatter(halocat['z'],halocat['m200'])

    filename_fig = 'figs/hist.cfhtlens.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' , filename_fig)


def select_halos_LRG():

    logger.info('getting BOSS galaxies overlapping with CFHTLens')

    range_z=config['range_z']
    range_M=config['range_M']
    filename_halos=config['filename_halos']

    filename_halos_cfhtlens = config['filename_BOSS_lrgs']
    halocat = pyfits.getdata(filename_halos_cfhtlens)
    logger.info('loaded %s with %d rows' % (filename_halos_cfhtlens,len(halocat)))

    # # select on Z
    select = (halocat['z'] > range_z[0]) * (halocat['z'] < range_z[1])
    halocat=halocat[select]
    logger.info('selected on Z number of halos: %d' % len(halocat))

    # select on proximity to CFHTLens
    logger.info('getting LRGs close to CFHTLens - Ball Tree for 2D')
    filename_cfhtlens_shears = config['filename_cfhtlens_shears']
    shearcat = tabletools.loadTable(filename_cfhtlens_shears)    
    if 'ALPHA_J2000' in shearcat.dtype.names:
        ra_field = 'ALPHA_J2000'
        de_field = 'DELTA_J2000'
    elif 'ra' in shearcat.dtype.names:
        ra_field = 'ra'
        de_field = 'dec'
        
    cfhtlens_coords = np.concatenate([shearcat[ra_field][:,None],shearcat[de_field][:,None]],axis=1)

    logger.info('getting BT')
    BT = BallTree(cfhtlens_coords, leaf_size=5)
    theta_add = 0.3
    boss_coords1 = np.concatenate([ halocat['ra'][:,None]           , halocat['dec'][:,None]           ] , axis=1)
    boss_coords2 = np.concatenate([ halocat['ra'][:,None]-theta_add , halocat['dec'][:,None]-theta_add ] , axis=1)
    boss_coords3 = np.concatenate([ halocat['ra'][:,None]+theta_add , halocat['dec'][:,None]-theta_add ] , axis=1)
    boss_coords4 = np.concatenate([ halocat['ra'][:,None]-theta_add , halocat['dec'][:,None]+theta_add ] , axis=1)
    boss_coords5 = np.concatenate([ halocat['ra'][:,None]+theta_add , halocat['dec'][:,None]+theta_add ] , axis=1)
    n_connections=1
    logger.info('getting neighbours')
    bt1_dx,bt_id = BT.query(boss_coords1,k=n_connections)
    bt2_dx,bt_id = BT.query(boss_coords2,k=n_connections)
    bt3_dx,bt_id = BT.query(boss_coords3,k=n_connections)
    bt4_dx,bt_id = BT.query(boss_coords4,k=n_connections)
    bt5_dx,bt_id = BT.query(boss_coords5,k=n_connections)
    limit_dx = 0.05
    select = (bt1_dx < limit_dx)*(bt2_dx < limit_dx)*(bt3_dx < limit_dx)*(bt4_dx < limit_dx)*(bt5_dx < limit_dx)
    select = select.flatten()
    halocat=halocat[select]

    perm3 = np.random.permutation(len(shearcat))[:20000]
    pl.figure(figsize=(50,30))
    pl.scatter(halocat['ra'] , halocat['dec'] , 70 , marker='s', c='g' )
    pl.scatter(shearcat[ra_field][perm3],shearcat[de_field][perm3] , 0.1  , marker='o', c='b')
    filename_fig = 'figs/scatter.lrgs_in_cfhtlens.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' % filename_fig)
    pl.close()

    logger.info('selected on proximity to CFHTLens, remaining number of halos: %d' % len(halocat))

    index=range(0,len(halocat))
    halocat = tabletools.ensureColumn(rec=halocat, name='id', arr=index, dtype='i8')
    halocat = tabletools.ensureColumn(rec=halocat, name='m200' )
    halocat = tabletools.ensureColumn(rec=halocat, name='snr'  )
    fix_case( halocat )
       
    tabletools.saveTable(filename_halos,halocat)
    logger.info('wrote %s' % filename_halos)

    pl.figure()
    pl.hist(halocat['z'],histtype='step',bins=200)
    filename_fig = 'figs/hist.cfhtlens.lrgs_redshifts.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' , filename_fig)

    pl.figure()
    pl.scatter(halocat['ra'],halocat['dec'])
    filename_fig = 'figs/scatter.cfhtlens.lrgs.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' , filename_fig)


def select_halos_random():

    range_z=config['range_z']
    range_M=config['range_M']
    filename_halos=config['filename_halos']

    logger.info('selecting halos in range_z (%2.2f,%2.2f) and range_M (%2.2e,%2.2e)' % (range_z[0],range_z[1],range_M[0],range_M[1]))

    filename_halos_cfhtlens = os.environ['HOME'] + '/data/CFHTLens/CFHTLenS_BOSSDR10-LRGs.fits'
    halocat = tabletools.loadTable(filename_halos_cfhtlens)

    sample = np.random.uniform(0,len(halocat),config['n_random_pairs'])
    sample = sample.astype('i4')

    halocat = halocat[sample]

    # select on proximity to CFHTLens
    logger.info('getting LRGs close to CFHTLens - Ball Tree for 2D')
    filename_cfhtlens_shears =  os.environ['HOME'] + '/data/CFHTLens/CFHTLens_2014-06-14.normalised.fits'
    shearcat = tabletools.loadTable(filename_cfhtlens_shears)    
    if 'ALPHA_J2000' in shearcat.dtype.names:
        ra_field = 'ALPHA_J2000'
        de_field = 'DELTA_J2000'
    elif 'ra' in shearcat.dtype.names:
        ra_field = 'ra'
        de_field = 'dec'

    cfhtlens_coords = np.concatenate([shearcat[ra_field][:,None],shearcat[de_field][:,None]],axis=1)

    perm = np.random.permutation(len(shearcat))[:config['n_random_pairs']]
    ra = shearcat['ALPHA_J2000'][perm]
    dec = shearcat['DELTA_J2000'][perm]

    halocat['ra'] = ra+np.random.randn(len(ra))*0.001
    halocat['dec'] = dec+np.random.randn(len(ra))*0.001
    
    logger.info('getting BT')
    BT = BallTree(cfhtlens_coords, leaf_size=5)
    theta_add = 0
    boss_coords1 = np.concatenate([ halocat['ra'][:,None]           , halocat['dec'][:,None]           ] , axis=1)
    boss_coords2 = np.concatenate([ halocat['ra'][:,None]-theta_add , halocat['dec'][:,None]-theta_add ] , axis=1)
    boss_coords3 = np.concatenate([ halocat['ra'][:,None]+theta_add , halocat['dec'][:,None]-theta_add ] , axis=1)
    boss_coords4 = np.concatenate([ halocat['ra'][:,None]-theta_add , halocat['dec'][:,None]+theta_add ] , axis=1)
    boss_coords5 = np.concatenate([ halocat['ra'][:,None]+theta_add , halocat['dec'][:,None]+theta_add ] , axis=1)
    n_connections=1
    logger.info('getting neighbours')
    bt1_dx,bt_id = BT.query(boss_coords1,k=n_connections)
    bt2_dx,bt_id = BT.query(boss_coords2,k=n_connections)
    bt3_dx,bt_id = BT.query(boss_coords3,k=n_connections)
    bt4_dx,bt_id = BT.query(boss_coords4,k=n_connections)
    bt5_dx,bt_id = BT.query(boss_coords5,k=n_connections)
    limit_dx = 0.1
    select = (bt1_dx < limit_dx)*(bt2_dx < limit_dx)*(bt3_dx < limit_dx)*(bt4_dx < limit_dx)*(bt5_dx < limit_dx)
    select = select.flatten()
    halocat=halocat[select]

    perm3 = np.random.permutation(len(shearcat))[:20000]
    pl.figure(figsize=(50,30))
    pl.scatter(halocat['ra']        , halocat['dec']        , 70 , marker='s', c='g' )
    pl.scatter(shearcat[ra_field][perm3],shearcat[de_field][perm3] , 0.1  , marker='o', c='b')
    filename_fig = 'figs/scatter.lrgs_in_cfhtlens.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' % filename_fig)
    pl.close()

    logger.info('selected on proximity to CFHTLens, remaining number of halos: %d' % len(halocat))

    index=range(0,len(halocat))
    halocat = tabletools.ensureColumn(rec=halocat, name='id', arr=index, dtype='i8')
    halocat = tabletools.ensureColumn(rec=halocat, name='m200' )
    halocat = tabletools.ensureColumn(rec=halocat, name='snr'  )
    fix_case( halocat )
    # halocat.dtype.names = ('field', 'index', 'ra', 'dec', 'z', 'snr', 'id')
        
    tabletools.saveTable(filename_halos,halocat)
    logger.info('wrote %s' % filename_halos)

    pl.figure()
    pl.hist(halocat['z'],histtype='step',bins=200)
    filename_fig = 'figs/hist.cfhtlens.lrgs_redshifts.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' , filename_fig)

def fit_halos():
    
    filename_halos = config['filename_halos']
    filename_shear = config['filename_shears']
    halos = tabletools.loadTable(filename_halos)
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='index',dtype='i4')
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='m200_fit',dtype='f4')
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='m200_sig',dtype='f4')
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='m200_errhi',dtype='f4')
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='m200_errlo',dtype='f4')
    n_halos = len(halos)
    logger.info('halos %d' , len(halos))

    id_first = args.first
    id_last = args.first + args.num
   
    box_size=30 # arcmin
    pixel_size=0.5
    vec_u_arcmin, vec_v_arcmin = np.arange(-box_size/2.,box_size/2.+1e-9,pixel_size), np.arange(-box_size/2.,box_size/2.+1e-9,pixel_size)
    grid_u_arcmin, grid_v_arcmin = np.meshgrid( vec_u_arcmin  , vec_v_arcmin ,indexing='ij')
    vec_u_rad , vec_v_rad   = cosmology.arcmin_to_rad(vec_u_arcmin,vec_v_arcmin)

    n_bins_tanglential = 20
    max_angle_tangential = 15
    tangential_profile_bins_arcmin = np.linspace(0,max_angle_tangential,n_bins_tanglential)
    tangential_profile_bins_rad = tangential_profile_bins_arcmin/60.*np.pi/180.
    stacked_tangential_profile = []
    stacked_halo_shear = []

    global cfhtlens_shear_catalog
    cfhtlens_shear_catalog=None
    logger.info('running on %d - %d',id_first,id_last)
    iall = 0
    for ih in range(id_first, id_last):
        iall+=1

        ihalo = halos['index'][ih]
        vhalo = halos[ih]

        if cfhtlens_shear_catalog == None:
            filename_cfhtlens_shears = config['filename_cfhtlens_shears']
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


        halo_z = vhalo['z']
        shear_bias_m = cfhtlens_shear_catalog['m']
        shear_weight = cfhtlens_shear_catalog['weight']

        halo_ra_rad , halo_de_rad = cosmology.deg_to_rad(vhalo['ra'],vhalo['dec'])
        shear_ra_rad , shear_de_rad = cosmology.deg_to_rad(shear_ra_deg, shear_de_deg)

        # get tangent plane projection        
        shear_u_rad, shear_v_rad = cosmology.get_gnomonic_projection(shear_ra_rad , shear_de_rad , halo_ra_rad , halo_de_rad)
        shear_g1_proj , shear_g2_proj = cosmology.get_gnomonic_projection_shear(shear_ra_rad , shear_de_rad , halo_ra_rad , halo_de_rad, shear_g1,shear_g2)       

        dtheta_x , dtheta_y = cosmology.arcmin_to_rad(box_size/2.,box_size/2.)
        select = ( np.abs( shear_u_rad ) < np.abs(dtheta_x)) * (np.abs(shear_v_rad) < np.abs(dtheta_y))

        shear_u_stamp_rad  = shear_u_rad[select]
        shear_v_stamp_rad  = shear_v_rad[select]
        shear_g1_stamp = shear_g1_proj[select]
        shear_g2_stamp = shear_g2_proj[select]
        shear_g1_orig = shear_g1[select]
        shear_g2_orig = shear_g2[select]
        shear_z_stamp = shear_z[select]
        shear_bias_m_stamp = shear_bias_m[select] 
        shear_weight_stamp = shear_weight[select]

        n_invalid = len(np.nonzero(shear_weight_stamp==0)[0])
        n_total = len(shear_weight_stamp)

        hist_g1, _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=shear_g1_stamp * shear_weight_stamp)
        hist_g2, _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=shear_g2_stamp * shear_weight_stamp)
        hist_n,  _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) )
        hist_m , _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=(1+shear_bias_m_stamp) * shear_weight_stamp )
        hist_w , _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=shear_weight_stamp)
        mean_g1 = hist_g1  / hist_m
        mean_g2 = hist_g2  / hist_m

        # in case we divided by zero
        hist_w[hist_n == 0] = 0
        hist_m[hist_n == 0] = 0
        hist_g1[hist_n == 0] = 0
        hist_g2[hist_n == 0] = 0
        mean_g1[hist_n == 0] = 0
        mean_g2[hist_n == 0] = 0

        select = np.isclose(hist_m,0)
        hist_w[select] = 0
        hist_m[select] = 0
        hist_g1[select] = 0
        hist_g2[select] = 0
        mean_g1[select] = 0
        mean_g2[select] = 0

        u_mid_arcmin, v_mid_arcmin = vec_u_arcmin[1:] - pixel_size/2. , vec_v_arcmin[1:] - pixel_size/2.
        grid_2d_u_arcmin , grid_2d_v_arcmin = np.meshgrid(u_mid_arcmin,v_mid_arcmin,indexing='ij')

        binned_u_arcmin = grid_2d_u_arcmin.flatten('F')
        binned_v_arcmin = grid_2d_v_arcmin.flatten('F')
        binned_g1 = mean_g1.flatten('F')
        binned_g2 = mean_g2.flatten('F')
        binned_n = hist_n.flatten('F')
        binned_w = hist_w.flatten('F')
 
        halo_shear = np.empty(len(binned_g1),dtype=dtype_halostamp)
        halo_shear['u_arcmin'] = binned_u_arcmin
        halo_shear['v_arcmin'] = binned_v_arcmin
        halo_shear['g1'] = binned_g1
        halo_shear['g2'] = binned_g2
        halo_shear['weight'] = binned_w
        halo_shear['n_gals'] = binned_n
        halo_shear['hist_g1'] = hist_g1.flatten('F')
        halo_shear['hist_g2'] = hist_g2.flatten('F')
        halo_shear['hist_m'] = hist_m.flatten('F')


        tabletools.savePickle(filename_shear,halo_shear,append=True,log=0)  
        import filaments_model_1h
        fitobj = filaments_model_1h.modelfit()
        fitobj.shear_u_arcmin =  halo_shear['u_arcmin']
        fitobj.shear_v_arcmin =  halo_shear['v_arcmin']
        fitobj.halo_u_arcmin = 0.
        fitobj.halo_v_arcmin = 0.
        fitobj.shear_g1 =  halo_shear['g1']
        fitobj.shear_g2 =  halo_shear['g2']
        fitobj.shear_w =  halo_shear['weight']
        fitobj.halo_z = vhalo['z']
        fitobj.get_bcc_pz(config['filename_pz'])
        fitobj.set_shear_sigma()
        fitobj.save_all_models=False

        fitobj.parameters[0]['box']['min'] = float(config['M200']['box']['min'])
        fitobj.parameters[0]['box']['max'] = float(config['M200']['box']['max'])
        fitobj.parameters[0]['n_grid'] = config['M200']['n_grid']

        log_post , grid_M200 = fitobj.run_gridsearch()
        ml_m200 = grid_M200[np.argmax(log_post)]
        prob_post = mathstools.normalise(log_post)
        max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_M200,prob_post)
        n_sig = max_par/err_lo

        halos['m200_fit'][ih]=ml_m200
        halos['m200_sig'][ih]= n_sig
        halos['m200_errhi'][ih]= err_hi
        halos['m200_errlo'][ih]= err_lo 

        pyfits.writeto(filename_halos,halos,clobber=True)
        logger.info('saved %s' , filename_halos)

        titlestr = '%5d n_gals=%d n_invalid=%d n_eff=%2.2f m200_fit=%2.2e +/- %2.2e %2.2e n_sig=%2.2f' % (ihalo,len(shear_g1_stamp),n_invalid,float(len(shear_weight_stamp))/(box_size**2),ml_m200,err_hi,err_lo,n_sig)
        logger.info(titlestr)

        if iall<100:
            pl.figure()
            pl.plot(grid_M200,prob_post)
            pl.title(titlestr,fontsize=8)
            filename_fig = 'figs/halofit.%05d.png' % ih
            pl.savefig(filename_fig)
            logger.info('saved %s',filename_fig)
            pl.close()

        gi = shear_g1_stamp+1j*shear_g2_stamp
        angle = np.angle(shear_u_stamp_rad+1j*shear_v_stamp_rad)
        dist = np.sqrt(shear_u_stamp_rad**2 + shear_v_stamp_rad**2)
        gt=-gi*np.exp(-1j*angle*2) 
        gt_err = shear_weight_stamp * (1+shear_bias_m_stamp)
        stacked_tangential_profile.append(np.array([dist,gt,gt_err]).T)

        g1 = shear_g1_stamp
        g2 = shear_g2_stamp
        u = shear_u_stamp_rad
        v = shear_v_stamp_rad
        w = shear_weight_stamp * (1+shear_bias_m_stamp)
        stacked_halo_shear.append(np.array([u,v,g1,g2,w]).T)


    logger.info('finished fitting %d individual halos',args.num)
    
    pl.figure()
    pl.hist(halos['m200_fit'],bins=50)
    pl.xlabel('m200_fit')
    filename_fig='figs/m200_fit_hist.png'
    pl.savefig(filename_fig)
    logger.info('saved %s',filename_fig)

    logger.info('getting tangential profile')
    stacked_tangential_profile = np.concatenate(stacked_tangential_profile,axis=0)
    print stacked_tangential_profile.shape
    d = stacked_tangential_profile[:,0]
    gt = stacked_tangential_profile[:,1]
    w = stacked_tangential_profile[:,2]
    binned_gt,_     = np.histogram(d,bins=tangential_profile_bins_rad,weights=gt*w)
    binned_gt_err,_ = np.histogram(d,bins=tangential_profile_bins_rad,weights=w)
    binned_gt = binned_gt/binned_gt_err
    binned_gt_err = np.sqrt(1./binned_gt_err)
    
    pl.figure()
    pl.errorbar(tangential_profile_bins_arcmin[1:],binned_gt,yerr=binned_gt_err,fmt='o')
    pl.xlabel('angle arcmin')
    pl.ylabel('g tangential')
    pl.title('using %d halos' % args.num)
    filename_fig = 'figs/tangential_lrgs.png'
    logger.info('saved %s',filename_fig)
    pl.savefig(filename_fig)
    pl.close()
    
    logger.info('fitting stack')
    stacked_halo_shear = np.concatenate(stacked_halo_shear,axis=0)
    print stacked_halo_shear.shape
    n_stacked_gals = stacked_halo_shear.shape[0]

    u=stacked_halo_shear[:,0]
    v=stacked_halo_shear[:,1]
    g1=stacked_halo_shear[:,2]
    g2=stacked_halo_shear[:,3]
    w=stacked_halo_shear[:,4]

    hist_g1, _, _    = np.histogram2d( x=u, y=v , bins=(vec_u_rad,vec_v_rad) , weights=g1 * w)
    hist_g2, _, _    = np.histogram2d( x=u, y=v , bins=(vec_u_rad,vec_v_rad) , weights=g2 * w)
    hist_n,  _, _    = np.histogram2d( x=u, y=v , bins=(vec_u_rad,vec_v_rad) )
    hist_w , _, _    = np.histogram2d( x=u, y=v , bins=(vec_u_rad,vec_v_rad) , weights=w)
    mean_g1 = hist_g1  / hist_w
    mean_g2 = hist_g2  / hist_w

    # in case we divided by zero
    hist_w[hist_n == 0] = 0
    hist_g1[hist_n == 0] = 0
    hist_g2[hist_n == 0] = 0
    mean_g1[hist_n == 0] = 0
    mean_g2[hist_n == 0] = 0

    select = np.isclose(hist_w,0)
    hist_w[select] = 0
    hist_g1[select] = 0
    hist_g2[select] = 0
    mean_g1[select] = 0
    mean_g2[select] = 0

    u_mid_arcmin, v_mid_arcmin = vec_u_arcmin[1:] - pixel_size/2. , vec_v_arcmin[1:] - pixel_size/2.
    grid_2d_u_arcmin , grid_2d_v_arcmin = np.meshgrid(u_mid_arcmin,v_mid_arcmin,indexing='ij')

    binned_u_arcmin = grid_2d_u_arcmin.flatten('F')
    binned_v_arcmin = grid_2d_v_arcmin.flatten('F')
    binned_g1 = mean_g1.flatten('F')
    binned_g2 = mean_g2.flatten('F')
    binned_n = hist_n.flatten('F')
    binned_w = hist_w.flatten('F')

    halo_shear = np.empty(len(binned_g1),dtype=dtype_halostamp)
    halo_shear['u_arcmin'] = binned_u_arcmin
    halo_shear['v_arcmin'] = binned_v_arcmin
    halo_shear['g1'] = binned_g1
    halo_shear['g2'] = binned_g2
    halo_shear['weight'] = binned_w
    halo_shear['n_gals'] = binned_n
    halo_shear['hist_g1'] = hist_g1.flatten('F')
    halo_shear['hist_g2'] = hist_g2.flatten('F')
    halo_shear['hist_m'] = hist_m.flatten('F')

    import filaments_model_1h
    fitobj = filaments_model_1h.modelfit()
    fitobj.shear_u_arcmin =  halo_shear['u_arcmin']
    fitobj.shear_v_arcmin =  halo_shear['v_arcmin']
    fitobj.halo_u_arcmin = 0.
    fitobj.halo_v_arcmin = 0.
    fitobj.shear_g1 =  halo_shear['g1']
    fitobj.shear_g2 =  halo_shear['g2']
    fitobj.shear_w =  halo_shear['weight']
    fitobj.halo_z = vhalo['z']
    fitobj.get_bcc_pz(config['filename_pz'])
    fitobj.set_shear_sigma()
    fitobj.save_all_models=False

    fitobj.parameters[0]['box']['min'] = float(config['M200']['box']['min'])
    fitobj.parameters[0]['box']['max'] = float(config['M200']['box']['max'])
    fitobj.parameters[0]['n_grid'] = config['M200']['n_grid']

    log_post , grid_M200 = fitobj.run_gridsearch()
    ml_m200 = grid_M200[np.argmax(log_post)]
    prob_post = mathstools.normalise(log_post)
    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_M200,prob_post)
    n_sig = max_par/err_lo

    halos['m200_fit'][ih]=ml_m200
    halos['m200_sig'][ih]= n_sig
    halos['m200_errhi'][ih]= err_hi
    halos['m200_errlo'][ih]= err_lo 

    logger.info('%5d n_gals=%d n_eff=%2.2f m200_fit=%2.2e +/- %2.2e %2.2e n_sig=%2.2f' % (ihalo,n_stacked_gals,n_stacked_gals/float((box_size**2)),ml_m200,err_hi,err_lo,n_sig))

    pl.figure()
    pl.plot(grid_M200,prob_post)
    filename_fig = 'figs/stack_halo_likelihood.png'
    pl.savefig(filename_fig)
    logger.info('saved %s', filename_fig)

    logger.info('plotting stack')
    n_box=len(vec_u_arcmin)-1
    u =np.reshape(halo_shear['u_arcmin'],[n_box,n_box],order='F')
    v =np.reshape(halo_shear['v_arcmin'],[n_box,n_box],order='F')
    g1=np.reshape(halo_shear['g1'],[n_box,n_box],order='F')
    g2=np.reshape(halo_shear['g2'],[n_box,n_box],order='F')
    ng=np.reshape(halo_shear['n_gals'],[n_box,n_box],order='F')
    sig=np.sqrt(1./np.reshape(halo_shear['weight'],[n_box,n_box],order='F'))
    sig_mask = np.ma.masked_invalid( sig )

    pl.figure(figsize=(10,10))
    pl.subplot(2,2,1)
    pl.pcolormesh(u,v,g1)
    pl.axis('tight')
    pl.title('g1')
    pl.colorbar()
    pl.subplot(2,2,2)
    pl.pcolormesh(u,v,g2)
    pl.axis('tight')
    pl.title('g2')
    pl.colorbar()
    pl.subplot(2,2,3)
    pl.pcolormesh(u,v,ng)
    pl.axis('tight')
    pl.title('n_gals')
    pl.colorbar()
    pl.subplot(2,2,4)
    cmap = pl.get_cmap('jet')
    cmap.set_bad('w')
    pl.pcolormesh(u,v,sig_mask,cmap=cmap)
    pl.axis('tight')
    pl.title('sig')
    pl.colorbar()
    pl.suptitle('stack with %d halos' % args.num)
    filename_fig = 'figs/stack_halo_image.png'
    pl.savefig(filename_fig)
    logger.info('saved %s', filename_fig)

    import pdb; pdb.set_trace()

def select_lrgs():

    select_fun = config['cfhtlens_select_fun']
    exec(select_fun+'()')

def main():

    valid_actions = ['select_lrgs','fit_halos']

    description = 'halo_stamps'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-f', '--first', default=0,type=int, action='store', help='first pair to process')
    parser.add_argument('-n', '--num', default=1,type=int, action='store', help='number of pairs to process')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )

    global args

    args = parser.parse_args()
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    logger.setLevel(logging_level)

    global config 
    config = yaml.load(open(args.filename_config))

    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    if args.actions==None:
        logger.error('no action specified, choose from %s' % valid_actions)
        return
    for action in valid_actions:
        if action in args.actions:
            logger.info('executing %s' % action)
            exec action+'()'
    for ac in args.actions:
        if ac not in valid_actions:
            print 'invalid action %s' % ac


    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


main()