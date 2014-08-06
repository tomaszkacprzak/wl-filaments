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

    range_z=map(float,config['range_z'])
    range_M=map(float,config['range_M'])
    filename_halos=config['filename_halos']


    import os
    import numpy as np
    import pylab as pl
    import tabletools
    import pyfits
    import plotstools

    filename_catalog_lrgs = config['filename_BOSS_lrgs']
    filename_catalog_clusters = config['filename_clusters']

    cat_clusters=tabletools.loadTable(filename_catalog_clusters)
    cat_lrgs=np.array(pyfits.getdata(filename_catalog_lrgs))

    select= ( (10**cat_clusters['m200'])>range_M[0] ) * (10**cat_clusters['m200']<range_M[1])
    cat_clusters=cat_clusters[select]

    coords_lrgs = np.concatenate([cat_lrgs['ra'][:,None],cat_lrgs['dec'][:,None]],axis=1)
    coords_clusters = np.concatenate([cat_clusters['ra'][:,None],cat_clusters['dec'][:,None]],axis=1)
    from sklearn.neighbors import BallTree as BallTree
    BT = BallTree(coords_lrgs, leaf_size=5)
    n_connections=1
    bt_dx,bt_id = BT.query(coords_clusters,k=n_connections)
    
    dx = 0.1
    select=bt_dx<dx
    ids_selected = bt_id[select]

    cat_lrgs_lrgsclus = cat_lrgs[ids_selected]
    cat_clus_lrgsclus = cat_clusters[select.flatten()]

    select = np.abs(cat_lrgs_lrgsclus['z'] - cat_clus_lrgsclus['z'])<0.2
    cat_lrgs_lrgsclus = cat_lrgs_lrgsclus[select]
    cat_clus_lrgsclus = cat_clus_lrgsclus[select]

    cat_join=cat_clus_lrgsclus.copy()
    cat_join['z'] = cat_lrgs_lrgsclus['z']
    print 'found %d lrg-clus matches' % len(cat_join)

    select= (cat_join['z'] > range_z[0]) * (cat_join['z'] < range_z[1])
    cat_join = cat_join[select]
    fix_case(cat_join)
    print 'selected on redshift range, n_clusters=%d' % len(cat_join)

    cat_join = tabletools.ensureColumn(name='m200_fit',rec=cat_join)
    cat_join = tabletools.ensureColumn(name='ra_clus',rec=cat_join)
    cat_join = tabletools.ensureColumn(name='dec_clus',rec=cat_join)
    cat_join['ra_clus'] = cat_clus_lrgsclus['ra']
    cat_join['dec_clus'] = cat_clus_lrgsclus['dec']
    cat_join['ra']  = cat_lrgs_lrgsclus['ra'].copy()
    cat_join['dec'] = cat_lrgs_lrgsclus['dec'].copy()
    

    pl.figure()
    pl.hist(cat_join['m200'])
    filename_fig = 'figs/hist.m200.png'
    pl.savefig(filename_fig)
    pl.close()
    print 'saved' , filename_fig

    pl.figure()
    pl.scatter(cat_lrgs['ra'],cat_lrgs['dec'],edgecolor='b',label='lrgs',s=100,facecolors='none')
    pl.scatter(cat_clusters['ra'],cat_clusters['dec'],edgecolor='r',label='clus',s=100,facecolors='none')
    pl.scatter(cat_lrgs_lrgsclus['ra'],cat_lrgs_lrgsclus['dec'],c='c',label='join lrg',s=20)
    pl.scatter(cat_clus_lrgsclus['ra'],cat_clus_lrgsclus['dec'],c='m',label='join clu',s=20)
    pl.legend()
    pl.show()

    import pdb; pdb.set_trace()
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


def randomise_halos():

    filename_halos=config['filename_halos']
    halocat = np.array(pyfits.getdata(filename_halos))

    sample = np.random.uniform(0,len(halocat),config['n_random_halos'])
    sample = sample.astype('i4')

    halocat = halocat[sample]

    # select on proximity to CFHTLens
    logger.info('getting LRGs close to CFHTLens - Ball Tree for 2D')
    filename_cfhtlens_shears =  config['filename_cfhtlens_shears']
    shearcat = tabletools.loadTable(filename_cfhtlens_shears)    
    if 'ALPHA_J2000' in shearcat.dtype.names:
        ra_field = 'ALPHA_J2000'
        de_field = 'DELTA_J2000'
    elif 'ra' in shearcat.dtype.names:
        ra_field = 'ra'
        de_field = 'dec'

    cfhtlens_coords = np.concatenate([shearcat[ra_field][:,None],shearcat[de_field][:,None]],axis=1)

    perm = np.random.permutation(len(shearcat))[:config['n_random_halos']]
    ra = shearcat[ra_field][perm]
    dec = shearcat[de_field][perm]

    halocat['ra'] = ra+np.random.randn(len(ra))*0.1
    halocat['dec'] = dec+np.random.randn(len(ra))*0.1
    
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

    halocat['index']=range(0,len(halocat))
    fix_case( halocat )

    filename_halos_random = filename_halos.replace('.fits','.random.fits')
    tabletools.saveTable(filename_halos_random,halocat)
    logger.info('wrote %s' % filename_halos_random)

    print "============================================"
    print " change filename_halos to filename_halos.random.fits"
    print "============================================"

def randomise_halos2():

    filename_halos=config['filename_halos']
    halocat = np.array(pyfits.getdata(filename_halos))
    range_M = map(float,config['range_M'])
    select = (halocat['m200_fit'] > range_M[0]) * (halocat['m200_fit'] < range_M[1])
    halocat=halocat[select]


    sample = np.random.uniform(0,len(halocat),config['n_random_halos']*config['n_resampling'])
    sample = sample.astype('i4')

    halocat = halocat[sample]
    halocat['z'] = np.linspace(0.25,0.4,len(halocat))
    Dsky_deg = 1
    min_deg_sep = 2

    # select on proximity to CFHTLens
    logger.info('loading shears')
    filename_cfhtlens_shears =  config['filename_cfhtlens_shears']
    shearcat = tabletools.loadTable(filename_cfhtlens_shears)    
    perm = np.random.permutation(len(shearcat))[:100000]
    shearcat = shearcat[perm]
    if 'ALPHA_J2000' in shearcat.dtype.names:
        ra_field = 'ALPHA_J2000'
        de_field = 'DELTA_J2000'
    elif 'ra' in shearcat.dtype.names:
        ra_field = 'ra'
        de_field = 'dec'

    shear_ra=shearcat[ra_field].copy()
    shear_de=shearcat[de_field].copy()
    del(shearcat)

    mid_points = []
    logger.info('n_halos=%d',len(halocat))

    for ir in range(config['n_resampling']):
        ra_unused = shear_ra.copy()
        de_unused = shear_de.copy()
        logger.info('resampling %d' % ir)

        for ih in range(0,config['n_random_halos'],2):

            perm = np.random.choice(len(ra_unused),1)

            mid_ra = ra_unused[perm] + np.random.randn()*0.01
            mid_de = de_unused[perm] + np.random.randn()*0.01

            rand_vec = Dsky_deg/2. * np.exp(1j*np.random.uniform(low=0, high=2*np.pi))

            halocat['ra'][ih]    = mid_ra + rand_vec.real
            halocat['dec'][ih]   = mid_de + rand_vec.imag
            halocat['ra'][ih+1]  = mid_ra - rand_vec.real
            halocat['dec'][ih+1] = mid_de - rand_vec.imag

            select = cosmology.get_angular_separation(mid_ra,mid_de,ra_unused,de_unused,unit='deg') > min_deg_sep
            ra_unused=ra_unused[select]
            de_unused=de_unused[select]

            logger.info('-- % 5d perm % 6d added mid point % 6.2f % 6.2f n_gals_left=% 6d' , ih, perm, mid_ra, mid_de , len(ra_unused))
            # logger.info('---h1 %2.4f %2.4f' , halocat['ra'][ih], halocat['dec'][ih])
            # logger.info('---h2 %2.4f %2.4f' , halocat['ra'][ih+1], halocat['dec'][ih+1])
            # logger.info('---remaining shear cat len %d' , len(ra_unused))
   
    pl.figure(figsize=(30,20))
    pl.scatter(halocat['ra']        , halocat['dec']        , 70 , marker='o', c= halocat['z'] )
    # pl.scatter(shearcat[ra_field][perm3],shearcat[de_field][perm3] , 0.1  , marker='o', c='b')
    filename_fig = 'figs/scatter.lrgs_in_cfhtlens.pdf'
    pl.savefig(filename_fig)
    logger.info('saved %s' % filename_fig)
    pl.show()
    pl.close()

    logger.info('selected on proximity to CFHTLens, remaining number of halos: %d' % len(halocat))

    halocat['index']=range(0,len(halocat))
    halocat['id']=range(0,len(halocat))


    filename_halos_random = filename_halos.replace('.fits','.random.fits')
    tabletools.saveTable(filename_halos_random,halocat)
    logger.info('wrote %s' % filename_halos_random)

    print "============================================"
    print " change filename_halos to filename_halos.random.fits"
    print "============================================"

def fit_halos():
    
    filename_halos = config['filename_halos']
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

    redshift_offset = 0.2

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
        select = ( np.abs( shear_u_rad ) < np.abs(dtheta_x)) * (np.abs(shear_v_rad) < np.abs(dtheta_y)) * (shear_z > (halo_z + redshift_offset))

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
        n_eff_this=float(len(shear_weight_stamp))/(box_size**2)
        n_gals_this = len(shear_g1_stamp)
        n_invalid_this = n_invalid

        titlestr = '%5d n_gals=%d n_invalid=%d n_eff=%2.2f m200_fit=%2.2e +/- %2.2e %2.2e n_sig=%2.2f' % (ihalo,len(shear_g1_stamp),n_invalid,float(len(shear_weight_stamp))/(box_size**2),ml_m200,err_hi,err_lo,n_sig)
        logger.info(titlestr)

        if args.legion:
            filename_halos_part = os.path.basename(filename_halos).replace('.fits','.%04d.pp2' % ih)
            line=np.array([ih,n_eff_this,n_gals_this,n_invalid_this,halos['m200_fit'][ih],halos['m200_sig'][ih],halos['m200_errhi'][ih],halos['m200_errlo'][ih]])
            res={'ml' : line, 'log_post' : log_post, 'grid_M200' : grid_M200 }
            tabletools.savePickle(filename_halos_part,res)
            continue

        pyfits.writeto(filename_halos,halos,clobber=True)
        logger.info('saved %s' , filename_halos)

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

    if args.legion:
	return 
    
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

def merge_legion():

    halos=tabletools.loadTable(config['filename_halos'])
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='index',dtype='i4')
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='m200_fit',dtype='f4')
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='m200_sig',dtype='f4')
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='m200_errhi',dtype='f4')
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='m200_errlo',dtype='f4')

    halos_like = np.zeros([len(halos),config['M200']['n_grid']])

    for ih in range(len(halos)):
            filename_halos_part = 'results_halos/'+os.path.basename(config['filename_halos']).replace('.fits','.%04d.pp2' % ih)
            res=tabletools.loadPickle(filename_halos_part)
            halos[ih]['m200_fit'] = res['ml'][4]
            halos[ih]['m200_sig'] = res['ml'][5]
            halos[ih]['m200_errhi'] = res['ml'][6]
            halos[ih]['m200_errlo'] = res['ml'][7]
            log_post = res['log_post'].copy()
            grid_M200 = res['grid_M200'].copy()

            frac_diff = np.diff(log_post[1:]) / np.diff(log_post[:-1])
            # frac_diff=np.append(0,np.diff(log_post,n=2))
            frac_diff=np.append(frac_diff,1)
            frac_diff=np.append(1,frac_diff)
            frac_diff=np.abs(frac_diff-1)
            select = frac_diff < 0.1
            import scipy.interpolate
            f = scipy.interpolate.interp1d(grid_M200[select],log_post[select])
            log_post[~select] = f(grid_M200[~select])
            # pl.plot(np.exp(log_post-log_post.max())); pl.plot(np.exp(res['log_post']-res['log_post'].max())); pl.show()

            halos_like[ih,:]=log_post

            # ih,n_eff_this,n_gals_this,n_invalid_this,halos['m200_fit'][ih],halos['m200_sig'][ih],halos['m200_errhi'][ih],halos['m200_errlo'][ih]]
    
    pyfits.writeto(config['filename_halos'],halos,clobber=True)
    logger.info('saved %s' , config['filename_halos'])

   
    prior_prob= np.sum(np.exp(halos_like - halos_like.max()),axis=0)
    downsample=1
    h,b = np.histogram(halos['m200_fit']/1e14,bins=grid_M200[::downsample*100]/1e14,normed=True)
    pl.plot(grid_M200[::downsample]/1e14,prior_prob[::downsample]/np.max(prior_prob))
    pl.plot(b[:-1],h/np.max(h))
    filename_fig = 'halos.prior.png'
    pl.savefig(filename_fig)
    logger.info('saved %s',filename_fig)
    pl.close()

    filename_prior = config['filename_halos'].replace('.fits','.prior.pp2')  
    prior_dict = {'halos_like':halos_like,'grid_M200':grid_M200[::downsample],'prior':np.log(prior_prob[::downsample])}
    tabletools.savePickle(filename_prior,prior_dict)

def add_closest_cluster():


    range_M=map(float,config['range_M'])
    filename_halos=config['filename_halos']

    import os
    import numpy as np
    import pylab as pl
    import tabletools
    import pyfits
    import plotstools

    cat_lrgs=tabletools.loadTable(config['filename_halos'])
    cat_clus=tabletools.loadTable(config['filename_cfhtlens_clusters'])
    cat_pair=tabletools.loadTable(config['filename_pairs'])
    tabletools.fixCase(cat_clus)

    coords_lrgs = np.concatenate([cat_lrgs['ra'][:,None],cat_lrgs['dec'][:,None]],axis=1)
    coords_clus = np.concatenate([cat_clus['ra'][:,None],cat_clus['dec'][:,None]],axis=1)
    from sklearn.neighbors import BallTree as BallTree
    BT = BallTree(coords_clus, leaf_size=5)
    n_connections=10
    bt_dx,bt_id = BT.query(coords_lrgs,k=n_connections)
    
    zerr=0.1
    ids_match=bt_id[:,0]
    for i in range(len(cat_lrgs)):
        select = (cat_clus[bt_id[i,:]]['z']-cat_lrgs['z'][i]) < zerr
        if sum(select) != 0:
            ids_match[i] = bt_id[i,select][0]
            print bt_dx[i,select] , cat_clus[bt_id[i,select]]['m200']
        else:
            select = (cat_clus[bt_id[i,:]]['z']-cat_lrgs['z'][i]) < zerr*10
            ids_match[i] = bt_id[i,select][0]

    ang_sep = cosmology.get_angular_separation(cat_lrgs['ra'],cat_lrgs['dec'],cat_clus['ra'][ids_match],cat_clus['dec'][ids_match],unit='deg')
    dist_mpc = ang_sep*cosmology.get_ang_diam_dist(cat_lrgs['z'])*ang_sep

    cat_clus_matched = cat_clus[ids_match]

    exclude = np.concatenate([cat_pair[cat_pair['analysis']==1]['ih1'] , cat_pair[cat_pair['analysis']==1]['ih2']])
    select= (dist_mpc<0.25) & (cat_lrgs['m200_fit'] > 5e13) 
    select[exclude] = False
    print 'number used to calibrate priors' , sum(select)

    print 'number of halos with prior at 2 mpc' , sum(dist_mpc[exclude] < 2)

    x = 10**cat_clus_matched['m200'][select]
    y = cat_lrgs['m200_fit'][select]
    s=(cat_lrgs['m200_errlo'][select] + cat_lrgs['m200_errhi'][select])/2.
    import fitting
    b,a,C=fitting.get_line_fit(x,y,s)
    print a
    print b
    print np.sqrt(C)
    cat_lrgs = tabletools.ensureColumn(name='m200_prior',rec=cat_lrgs)
    cat_lrgs['m200_prior'] =  (10**cat_clus_matched['m200'])*a + b


    pl.figure()
    pl.errorbar(10**cat_clus_matched['m200'][exclude],cat_lrgs['m200_fit'][exclude],yerr=[cat_lrgs['m200_errlo'][exclude],cat_lrgs['m200_errhi'][exclude]],fmt='.')
    cax=pl.scatter(10**cat_clus_matched['m200'][exclude],cat_lrgs['m200_fit'][exclude],c=dist_mpc[exclude],marker='o',s=100)
    pl.colorbar(cax)
    # xmin,xmax = pl.xlim()
    # pl.plot(np.linspace(xmin,xmax,1000),np.linspace(xmin,xmax,1000)*(1+a)+b)
    pl.title('filament sample')
    pl.xlabel('cluster mass')
    pl.ylabel('lensing mass')
    pl.axis('equal')


    pl.figure()
    # pl.errorbar(10**cat_clus_matched['m200'][select],cat_lrgs['m200_fit'][select],yerr=[cat_lrgs['m200_errlo'][select],cat_lrgs['m200_errhi'][select]],fmt='.')
    cax=pl.scatter(10**cat_clus_matched['m200'][select],cat_lrgs['m200_fit'][select],c=dist_mpc[select],marker='o',s=100)
    pl.colorbar(cax)
    # xmin,xmax = pl.xlim()
    # pl.plot(np.linspace(xmin,xmax,1000),np.linspace(xmin,xmax,1000)*(1+a)+b)
    pl.title('rest of the sample')
    pl.xlabel('cluster mass')
    pl.ylabel('lensing mass')
    pl.axis('equal')


    pl.figure()
    pl.hist(dist_mpc,100)

    pl.show()

    import pdb; pdb.set_trace()




def main():

    valid_actions = ['select_lrgs','fit_halos','merge_legion','add_closest_cluster','randomise_halos','randomise_halos2']

    description = 'halo_stamps'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-f', '--first', default=0,type=int, action='store', help='first pair to process')
    parser.add_argument('-n', '--num', default=1,type=int, action='store', help='number of pairs to process')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )
    parser.add_argument('--legion', action='store_true', help='if cluster run')

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
