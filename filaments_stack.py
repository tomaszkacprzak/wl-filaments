import sys, os, logging, yaml, argparse, time
import pylab as pl
import numpy as np
import tabletools, plotstools, mathstools, cosmology
import filaments_tools
import filaments_model_2hf
import filaments_model_1h

logging_level = logging.INFO
logger = logging.getLogger("fil..stack") 
logger.setLevel(logging_level)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s", "%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False

def stack_halos():
    
    filename_pairs = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')
    filename_halos = config['filename_halos']
    halos = tabletools.loadTable(filename_halos)
    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)

    id_first = args.first
    id_last = args.first + args.num
    idh1 = pairs[id_first:id_last]['ih1']
    idh2 = pairs[id_first:id_last]['ih2']

    halos = halos[idh2]

    mass_cut = 14.0
    select = halos['m200'] > mass_cut
    halos=halos[select]
    # halos=halos[7:8]
    # print 'n halos > %f %d' %( mass_cut,len(halos) )
    # print halos['m200']

    print "==============================="
    print 'pair id' , args.first
    print 'halo[m200]' , halos['m200']
    print 'ih2' , idh2
    print "==============================="

    box_size=40 # arcmin
    pixel_size=0.5
    vec_u_arcmin, vec_v_arcmin = np.arange(-box_size/2.,box_size/2.+1e-9,pixel_size), np.arange(-box_size/2.,box_size/2.+1e-9,pixel_size)
    grid_u_arcmin, grid_v_arcmin = np.meshgrid(plotstools.get_bins_centers(vec_u_arcmin) , plotstools.get_bins_centers(vec_v_arcmin),indexing='ij')
    vec_u_rad , vec_v_rad   = cosmology.arcmin_to_rad(vec_u_arcmin,vec_v_arcmin)
    grid_u_rad , grid_v_rad = cosmology.arcmin_to_rad(grid_u_arcmin,grid_v_arcmin)
    binned_shear_g1 = np.zeros_like(grid_u_arcmin)
    binned_shear_g2 = np.zeros_like(grid_u_arcmin) 
    binned_shear_n  = np.zeros_like(grid_u_arcmin) 
    binned_shear_w  = np.zeros_like(grid_u_arcmin) 
    binned_shear_m  = np.zeros_like(grid_u_arcmin) 

    cfhtlens_shear_catalog=None
    for ihalo,vhalo in enumerate([halos]):

        global cfhtlens_shear_catalog
        if cfhtlens_shear_catalog == None:
            filename_chftlens_shears = os.environ['HOME']+ '/data/CFHTLens/CFHTLens_2014-04-07.fits'
            cfhtlens_shear_catalog = tabletools.loadTable(filename_chftlens_shears)
            if 'star_flag' in cfhtlens_shear_catalogger.dtype.names:
                select = cfhtlens_shear_catalog['star_flag'] == 0
                cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
                select = cfhtlens_shear_catalog['fitclass'] == 0
                cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
                logger.info('removed stars, remaining %d' , len(cfhtlens_shear_cahalos))

                select = (cfhtlens_shear_catalog['e1'] != 0.0) * (cfhtlens_shear_catalog['e2'] != 0.0)
                cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
                logger.info('removed zeroed shapes, remaining %d' , len(cfhtlens_shear_catalog))


        halo_z = vhalo['z']
        shear_bias_m = cfhtlens_shear_catalog['m']
        shear_weight = cfhtlens_shear_catalog['weight']

        # correcting additive systematics
        shear_g1 , shear_g2 = cfhtlens_shear_catalog['e1'] , -(cfhtlens_shear_catalog['e2']  - cfhtlens_shear_catalog['c2'])
        shear_ra_deg , shear_de_deg , shear_z = cfhtlens_shear_catalog['ra'] , cfhtlens_shear_catalog['dec'] ,  cfhtlens_shear_catalog['z']

        halo_ra_rad , halo_de_rad = cosmology.deg_to_rad(vhalo['ra'],vhalo['dec'])
        shear_ra_rad , shear_de_rad = cosmology.deg_to_rad(shear_ra_deg, shear_de_deg)

        # get tangent plane projection        
        shear_u_rad, shear_v_rad = cosmology.get_gnomonic_projection(shear_ra_rad , shear_de_rad , halo_ra_rad , halo_de_rad   )
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

        hist_g1, _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=shear_g1_stamp * shear_weight_stamp)
        hist_g2, _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=shear_g2_stamp * shear_weight_stamp)
        hist_n,  _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) )
        hist_m , _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=(1+shear_bias_m_stamp) * shear_weight_stamp )
        hist_w , _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=shear_weight_stamp)
        mean_g1 = hist_g1  / hist_w
        mean_g2 = hist_g2  / hist_w

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

        binned_shear_g1 += hist_g1
        binned_shear_g2 += hist_g2
        binned_shear_n += hist_n
        binned_shear_w += hist_w
        binned_shear_m += hist_m

        logger.info(ihalo)


    binned_shear_g1 /= binned_shear_m
    binned_shear_g2 /= binned_shear_m
    

    pl.figure()
    pl.subplot(1,2,1)
    pl.imshow(binned_shear_g1,interpolation='nearest')
    pl.colorbar()

    pl.subplot(1,2,2)
    pl.imshow(binned_shear_g2,interpolation='nearest')
    pl.colorbar()

    pl.savefig('stacked_halos.png')
    logger.info('saved stacked_halos.png')
    pl.close()

    # pl.show()

    pickle_dict={}
    pickle_dict['binned_shear_g1'] = binned_shear_g1
    pickle_dict['binned_shear_g2'] = binned_shear_g2
    pickle_dict['binned_shear_n'] = binned_shear_n
    pickle_dict['binned_shear_w'] = binned_shear_w
    pickle_dict['grid_u_arcmin'] = grid_u_arcmin
    pickle_dict['grid_v_arcmin'] = grid_v_arcmin
    pickle_dict['mean_z'] = np.mean(halos['z'])
    tabletools.savePickle('stacked_halos.pp2',pickle_dict)

def fit_stacked_halos():

    pickle_dict=tabletools.loadPickle('stacked_halos.pp2')

    fitobj = filaments_model_1h.modelfit()
    fitobj.get_bcc_pz(filename_lenscat=config['filename_pz'])
    fitobj.shear_g1 =  pickle_dict['binned_shear_g1'].flatten()
    fitobj.shear_g2 =  pickle_dict['binned_shear_g2'].flatten()
    fitobj.shear_u_arcmin =  pickle_dict['grid_u_arcmin'].flatten()
    fitobj.shear_v_arcmin =  pickle_dict['grid_v_arcmin'].flatten()
    fitobj.halo_u_arcmin =  0
    fitobj.halo_v_arcmin =  0
    fitobj.halo_z = pickle_dict['mean_z']
    fitobj.save_all_models=False
    fitobj.shear_n_gals = pickle_dict['binned_shear_n'].flatten('F')
    fitobj.shear_w = pickle_dict['binned_shear_w'].flatten('F')
    fitobj.set_shear_sigma()
    logger.info('using different sigma_g per pixel mean(inv_sq_sigma_g)=%2.5f len(inv_sq_sigma_g)=%d' , np.mean(fitobj.inv_sq_sigma_g) , len(fitobj.inv_sq_sigma_g))


    logger.info('running grid search')
    log_post , grid_M200, best_g1, best_g2, best_limit_mask = fitobj.run_gridsearch(M200_min=12.5,M200_max=16,M200_n=100)
    prob_post = mathstools.normalise(log_post)
    pl.figure()
    pl.plot(grid_M200 , log_post , '.-')
    pl.figure()
    pl.plot(grid_M200 , prob_post , '.-')
    
    fitobj.plot_shears_all(fitobj.shear_g1,fitobj.shear_g2)
    fitobj.plot_shears_all(best_g1,best_g2)
    
    # pl.figure()
    # fitobj.plot_residual(best_g1, best_g2)
    

def stack_pairs():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_halos']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    scaling = np.mean(pairs['Dxy'])/2.
    print scaling

    vec_u_mpc, vec_v_mpc = np.arange(-2,2,0.05)*scaling, np.arange(-2,2,0.05)*scaling
    grid_u_mpc, grid_v_mpc = np.meshgrid(plotstools.get_bins_centers(vec_u_mpc) , plotstools.get_bins_centers(vec_v_mpc),indexing='ij')
    shear_g1 = np.zeros_like(grid_u_mpc)
    shear_g2 = np.zeros_like(grid_u_mpc) 
    shear_n  = np.zeros_like(grid_u_mpc) 
    shear_w  = np.zeros_like(grid_u_mpc) 

    halo1_u = scaling
    halo1_v = 0
    halo2_u = -scaling
    halo2_v = 0

    n_used=0
    list_z = []

    # pairs = tabletools.appendColumn(rec=pairs,arr=np.zeros(len(pairs)),name='area_arcmin2',dtype='f4')
    # pairs = tabletools.appendColumn(rec=pairs,arr=np.zeros(len(pairs)),name='n_eff',dtype='f4')

    id_pair_first = args.first
    id_pair_last = len(pairs) if args.num==-1 else args.first+args.num
    if id_pair_first>=id_pair_last: raise Exception('check -f and -n')

    for id_pair in range(id_pair_first,id_pair_last):
        
        shears_info = tabletools.loadPickle(config['filename_shears'],pos=id_pair)     
        mean_mass=(halo1['m200'][id_pair] + halo2['m200'][id_pair])/2

        print "==============================="
        print 'id_pair' , id_pair
        print 'halo1[m200]' , halo1['m200_fit'][id_pair]
        print 'halo2[m200]' , halo2['m200_fit'][id_pair]
        print 'total n_gals' , np.sum(shears_info['n_gals'])
        print "==============================="

        if (halo1['m200_fit'][id_pair] < 5e13) | (halo2['m200_fit'][id_pair] < 5e13):
            continue

        shears_info['u_mpc'] = shears_info['u_mpc'] / pairs['u1_mpc'][id_pair] * scaling
        shears_info['v_mpc'] = shears_info['v_mpc'] / pairs['u1_mpc'][id_pair] * scaling
        pairs_current = pairs[id_pair].copy()
        pairs_current['u1_mpc'] = pairs['u1_mpc'][id_pair] / pairs['u1_mpc'][id_pair] * scaling
        pairs_current['v1_mpc'] = pairs['v1_mpc'][id_pair] / pairs['u1_mpc'][id_pair] * scaling
        pairs_current['u2_mpc'] = pairs['u2_mpc'][id_pair] / pairs['u1_mpc'][id_pair] * scaling
        pairs_current['v2_mpc'] = pairs['v2_mpc'][id_pair] / pairs['u1_mpc'][id_pair] * scaling

        hist_g1, _, _    = np.histogram2d( x=shears_info['u_mpc'], y=shears_info['v_mpc'] , bins=(vec_u_mpc,vec_v_mpc) , weights=shears_info['g1']*shears_info['weight'])
        hist_g2, _, _    = np.histogram2d( x=shears_info['u_mpc'], y=shears_info['v_mpc'] , bins=(vec_u_mpc,vec_v_mpc) , weights=shears_info['g2']*shears_info['weight'])
        hist_w,  _, _    = np.histogram2d( x=shears_info['u_mpc'], y=shears_info['v_mpc'] , bins=(vec_u_mpc,vec_v_mpc) , weights=shears_info['weight'])
        hist_n,  _, _    = np.histogram2d( x=shears_info['u_mpc'], y=shears_info['v_mpc'] , bins=(vec_u_mpc,vec_v_mpc) , weights=shears_info['n_gals'])

        hist_g1[np.isnan(hist_g1)]=0
        hist_g2[np.isnan(hist_g2)]=0
        shear_g1 += hist_g1
        shear_g2 += hist_g2
        shear_n += hist_n
        shear_w += hist_w


        list_z.append(pairs[id_pair]['z'])
        n_used +=1
        logger.info('stacked pair %d n_used %d' , id_pair , n_used)

    mean_g1 = shear_g1  / shear_w
    mean_g2 = shear_g2  / shear_w
    mean_z = np.mean(list_z)
        
    filaments_tools.plot_pair(halo1_u,halo1_v,halo2_u,halo2_v,grid_u_mpc.flatten(),grid_v_mpc.flatten(),mean_g1.flatten(),mean_g2.flatten(),idp=0,nuse=10000,filename_fig=None,show=False,close=False,halo1=None,halo2=None,pair_info=None,quiver_scale=1,plot_type='quiver')
    pl.show()

    # fit stack
    fitobj = filaments_model_2hf.modelfit()
    fitobj.get_bcc_pz(config['filename_pz'])
    prob_z = fitobj.prob_z
    fitobj.shear_u_arcmin =  grid_u_mpc.flatten()
    fitobj.shear_v_arcmin =  grid_v_mpc.flatten()
    fitobj.shear_u_mpc =  grid_u_mpc.flatten()
    fitobj.shear_v_mpc =  grid_v_mpc.flatten()

    fitobj.shear_g1 =  mean_g1.flatten()
    fitobj.shear_g2 =  mean_g2.flatten()

    fitobj.shear_n_gals = shear_n.flatten()
    fitobj.shear_w =  shear_w.flatten()
    fitobj.set_shear_sigma()
    logger.info('using different sigma_g per pixel mean(inv_sq_sigma_g)=%2.5f len(inv_sq_sigma_g)=%d' , np.mean(fitobj.inv_sq_sigma_g) , len(fitobj.inv_sq_sigma_g))

    
    fitobj.halo1_u_arcmin =  halo1_u
    fitobj.halo1_v_arcmin =  halo1_v
    fitobj.halo1_u_mpc =  halo1_u
    fitobj.halo1_v_mpc =  halo1_v
    fitobj.halo1_z =  mean_z

    fitobj.halo2_u_arcmin =  halo2_u
    fitobj.halo2_v_arcmin =  halo2_v
    fitobj.halo2_u_mpc = halo2_u
    fitobj.halo2_v_mpc = halo2_v
    fitobj.halo2_z = mean_z

    fitobj.parameters[0]['box']['min'] = config['kappa0']['box']['min']
    fitobj.parameters[0]['box']['max'] = config['kappa0']['box']['max']
    fitobj.parameters[0]['n_grid'] = config['kappa0']['n_grid']

    fitobj.parameters[1]['box']['min'] = config['radius']['box']['min']
    fitobj.parameters[1]['box']['max'] = config['radius']['box']['max']
    fitobj.parameters[1]['n_grid'] = config['radius']['n_grid']
    
    fitobj.parameters[2]['box']['min'] = float(config['h1M200']['box']['min'])
    fitobj.parameters[2]['box']['max'] = float(config['h1M200']['box']['max'])
    fitobj.parameters[2]['n_grid'] = config['h1M200']['n_grid']
    
    fitobj.parameters[3]['box']['min'] = float(config['h2M200']['box']['min'])
    fitobj.parameters[3]['box']['max'] = float(config['h2M200']['box']['max'])
    fitobj.parameters[3]['n_grid'] = config['h2M200']['n_grid']

    filename_results_grid = 'stack.grid.pp2'
    filename_results_prob = 'stack.prob.pp2'

    if config['optimization_mode'] == 'gridsearch':
            logger.info('running grid search')

            log_post , params, grids = fitobj.run_gridsearch()
            tabletools.savePickle(filename_results_prob,log_post.astype(np.float32))
            prob_post , _ , _ , _ = mathstools.get_normalisation(log_post)
            
            grid_info = {}
            grid_info['grid_kappa0'] = grids[0]
            grid_info['grid_radius'] = grids[1]
            grid_info['grid_h1M200'] = grids[2]
            grid_info['grid_h2M200'] = grids[3]
            grid_info['post_kappa0'] = params[0]
            grid_info['post_radius'] = params[1]
            grid_info['post_h1M200'] = params[2]
            grid_info['post_h2M200'] = params[3]
            tabletools.savePickle(filename_results_grid,grid_info)

            X=[grid_info['grid_kappa0'],grid_info['grid_radius'],grid_info['grid_h1M200'],grid_info['grid_h2M200']]
            plotstools.plot_dist_meshgrid(X,prob_post)
            

def plot_stack_fit():

    filename_results_grid = 'stack.grid.pp2'
    filename_results_prob = 'stack.prob.pp2'

    log_prob=tabletools.loadPickle(filename_results_prob)
    grid=tabletools.loadPickle(filename_results_grid)

    grid['grid_h1M200'] = grid['grid_h1M200'][:,:,12:25,1:10]
    grid['grid_h2M200'] = grid['grid_h2M200'][:,:,12:25,1:10]
    grid['grid_kappa0'] = grid['grid_kappa0'][:,:,12:25,1:10]
    grid['grid_radius'] = grid['grid_radius'][:,:,12:25,1:10]
    log_prob = log_prob[:,:,12:25,1:10]

    X=[grid['grid_kappa0'],grid['grid_radius'],grid['grid_h1M200'],grid['grid_h2M200']]
    prob = np.exp(log_prob-log_prob.max())
    mdd = plotstools.multi_dim_dist()
    mdd.plot_dist_meshgrid(X,prob)

    filename_fig = 'figs/stack_triangle.png'
    pl.savefig(filename_fig)
    pl.show()
    pl.close()
    logger.info('saved %s',filename_fig)




    import pdb; pdb.set_trace()

def stack_pairs_new():

    redshift_offset = 0.2

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_halos']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    import graphstools
    select_best=graphstools.get_best_neighbour(pairs,halo1,halo2)
    # select_best = [1,2]
    
    halo1 = halo1[select_best]
    halo2 = halo2[select_best]
    pairs = pairs[select_best]

    logger.info('loading shear catalog')
    import fitsio
    columns = ['e1', 'e2', 'm', 'c2', 'z', 'ra' , 'dec' , 'star_flag' , 'fitclass' , 'weight']
    cfhtlens_shear_catalog = np.array(fitsio.read(config['filename_cfhtlens_shears'], columns=columns))
    logger.info('size of shear catalog %2.2f MB' % (cfhtlens_shear_catalog.nbytes/1000000.))

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
        shear_weight = cfhtlens_shear_catalog['weight'] 
        shear_bias_m = cfhtlens_shear_catalog['m']

    all_shear_u_stamp_rad = []
    all_shear_v_stamp_rad = []
    all_shear_u_stamp_mpc = []
    all_shear_v_stamp_mpc = []
    all_shear_g1_stamp = []
    all_shear_g2_stamp = []
    all_shear_g1_orig = []
    all_shear_g2_orig = []
    all_shear_z_stamp = []
    all_shear_ra_stamp_deg = []
    all_shear_de_stamp_deg = []
    all_shear_bias_m_stamp = []
    all_shear_weight_stamp = []
    all_g1_sc = []
    all_g2_sc = []
    all_shear_sc = []

    n_gals_in_stack = 0

    # iterate over pairs
    for ipair,vpair in enumerate(pairs):

        logger.info('========= pair %03d ==========' % ipair)

        h1 = halo1[ipair]
        h2 = halo2[ipair]
        
        halo1_ra_deg , halo1_de_deg = h1['ra'],h1['dec']
        halo2_ra_deg , halo2_de_deg = h2['ra'],h2['dec']

        pair_ra_deg,  pair_de_deg = cosmology.get_midpoint(halo1_ra_deg , halo1_de_deg , halo2_ra_deg , halo2_de_deg,unit='deg')
        pair_z = np.mean([h1['z'],h2['z']])

        # convert to radians
        halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(halo1_ra_deg, halo1_de_deg)
        halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(halo2_ra_deg, halo2_de_deg)
        shear_ra_rad , shear_de_rad = cosmology.deg_to_rad(shear_ra_deg, shear_de_deg)

        # import pylab as pl
        # pl.figure()
        # pl.scatter(halo1_ra_rad     , halo1_de_rad , 100, 'c' , marker='o')
        # pl.scatter(halo2_ra_rad     , halo2_de_rad ,100,  'm' , marker='o')
        # select = np.random.permutation(len(shear_ra_rad))[:10000]
        # pl.scatter(shear_ra_rad[select]  , shear_de_rad[select] ,1,  'm' , marker='.')
        # pl.show()
        # ang_sep = cosmology.get_angular_separation(halo1_ra_rad, halo1_de_rad, halo2_ra_rad, halo2_de_rad)
        # print 'angular sep: ' , ang_sep , ang_sep * cosmology.get_ang_diam_dist(pair_z)

        # get midpoint
        pairs_ra_rad , pairs_de_rad = cosmology.get_midpoint_rad(halo1_ra_rad,halo1_de_rad,halo2_ra_rad,halo2_de_rad)

        # get tangent plane projection        
        shear_u_rad, shear_v_rad = cosmology.get_gnomonic_projection(shear_ra_rad , shear_de_rad , pairs_ra_rad , pairs_de_rad)
        pairs_u_rad, pairs_v_rad = cosmology.get_gnomonic_projection(pairs_ra_rad , pairs_de_rad , pairs_ra_rad , pairs_de_rad)
        halo1_u_rad, halo1_v_rad = cosmology.get_gnomonic_projection(halo1_ra_rad , halo1_de_rad , pairs_ra_rad , pairs_de_rad)
        halo2_u_rad, halo2_v_rad = cosmology.get_gnomonic_projection(halo2_ra_rad , halo2_de_rad , pairs_ra_rad , pairs_de_rad)
        shear_g1_proj , shear_g2_proj = cosmology.get_gnomonic_projection_shear(shear_ra_rad , shear_de_rad , pairs_ra_rad , pairs_de_rad, shear_g1,shear_g2)

        rotation_angle = np.angle(halo1_u_rad + 1j*halo1_v_rad)

        shear_u_rot_rad , shear_v_rot_rad = filaments_tools.rotate_vector(rotation_angle, shear_u_rad , shear_v_rad)
        halo1_u_rot_rad , halo1_v_rot_rad = filaments_tools.rotate_vector(rotation_angle, halo1_u_rad , halo1_v_rad)
        halo2_u_rot_rad , halo2_v_rot_rad = filaments_tools.rotate_vector(rotation_angle, halo2_u_rad , halo2_v_rad)   
        shear_g1_rot , shear_g2_rot = filaments_tools.rotate_shear(rotation_angle, shear_u_rad, shear_v_rad, shear_g1_proj, shear_g2_proj)


        # import pylab as pl
        # pl.figure()
        # select = np.random.permutation(len(shear_v_rot_rad))[:10000]
        # pl.scatter( shear_u_rot_rad[select] ,shear_v_rot_rad[select] , 1 , marker=',')
        # pl.scatter( 0     , 0 , 100, 'r' , marker='x')
        # pl.scatter(halo1_u_rot_rad , halo1_v_rot_rad , 100, 'c' , marker='o')
        # pl.scatter(halo2_u_rot_rad , halo2_v_rot_rad , 100, 'm' , marker='o')
        # pl.show()
        # ang_sep = cosmology.get_angular_separation(halo1_u_rot_rad, halo1_v_rot_rad, halo2_u_rot_rad, halo2_v_rot_rad)
        # print 'angular sep: ' , ang_sep , ang_sep * cosmology.get_ang_diam_dist(pair_z)


        # grid boudaries

        dtheta_x = config['boundary_mpc'] / cosmology.get_ang_diam_dist(pair_z) 
        dtheta_y = config['boundary_mpc'] / cosmology.get_ang_diam_dist(pair_z) 

        # select = (shear_v_rot_rad < dtheta_y) * (shear_v_rot_rad > -dtheta_y) * (shear_u_rot_rad < (halo1_u_rot_rad + dtheta_x)) *  (shear_u_rot_rad > (halo2_u_rot_rad - dtheta_x))
        select = ( np.abs( shear_u_rot_rad ) < np.abs(halo1_u_rot_rad + dtheta_x)) * (np.abs(shear_v_rot_rad) < np.abs(dtheta_y)) * (shear_z > (pair_z + redshift_offset))
        n_gals_in_this_stamp = sum(select)
        n_gals_in_stack += n_gals_in_this_stamp
        logger.info('selected %d galaxies for that stamp, total galaxies %d' % (n_gals_in_this_stamp,n_gals_in_stack))

        range_u_mpc , range_v_mpc = cosmology.rad_to_mpc(2*halo1_u_rot_rad+2*dtheta_x,2*dtheta_y,pair_z)
        range_u_arcmin , range_v_arcmin = cosmology.rad_to_arcmin(2*halo1_u_rot_rad+2*dtheta_x,2*dtheta_y)

        logger.info('z=%f dtheta=%f' % (pair_z,dtheta_x))
        logger.info('range_u_mpc=%2.4f range_v_mpc=%2.4f' % (range_u_mpc , range_v_mpc) )

        # select
        shear_u_stamp_rad = shear_u_rot_rad[select]
        shear_v_stamp_rad = shear_v_rot_rad[select]
        shear_g1_stamp = shear_g1_rot[select]
        shear_g2_stamp = shear_g2_rot[select]
        shear_z_stamp = shear_z[select]
    
        # convert to Mpc   
        shear_u_stamp_mpc , shear_v_stamp_mpc = cosmology.rad_to_mpc(shear_u_stamp_rad,shear_v_stamp_rad,pair_z)
        halo1_u_rot_mpc , halo1_v_rot_mpc = cosmology.rad_to_mpc(halo1_u_rot_rad,halo1_v_rot_rad,pair_z)
        halo2_u_rot_mpc , halo2_v_rot_mpc = cosmology.rad_to_mpc(halo2_u_rot_rad,halo2_v_rot_rad,pair_z)

        shear_u_stamp_arcmin , shear_v_stamp_arcmin = cosmology.rad_to_arcmin(shear_u_stamp_rad,shear_v_stamp_rad)
        halo1_u_rot_arcmin , halo1_v_rot_arcmin = cosmology.rad_to_arcmin(halo1_u_rot_rad,halo1_v_rot_rad)
        halo2_u_rot_arcmin , halo2_v_rot_arcmin = cosmology.rad_to_arcmin(halo2_u_rot_rad,halo2_v_rot_rad)

        # scale    
        scale = 1
        shear_scaled_u = scale*shear_u_stamp_mpc
        shear_scaled_v = scale*shear_v_stamp_mpc

        sc = cosmology.get_sigma_crit( shear_z_stamp , np.ones(shear_z_stamp.shape)*pair_z , unit=config['Sigma_crit_unit'] )      
        scinv = 1./sc
        g1sc = shear_g1_stamp * sc
        g2sc = shear_g2_stamp * sc

        # select the stamp
        all_shear_u_stamp_mpc.append(  shear_scaled_u )
        all_shear_v_stamp_mpc.append(  shear_scaled_v )
        all_shear_g1_stamp.append( shear_g1_stamp )
        all_shear_g2_stamp.append( shear_g2_stamp )
        all_shear_g1_orig.append(      shear_g1[select] )
        all_shear_g2_orig.append(      shear_g2[select] )
        all_g1_sc.append( g1sc )
        all_g2_sc.append( g2sc )
        all_shear_z_stamp.append(      shear_z[select] )
        all_shear_ra_stamp_deg.append( shear_ra_deg[select] )
        all_shear_de_stamp_deg.append( shear_de_deg[select] )
        all_shear_bias_m_stamp.append( shear_bias_m[select] )
        all_shear_weight_stamp.append( shear_weight[select] )
        all_shear_sc.append(sc)



    # end loop    
    shear_u_stamp_mpc  = np.concatenate(all_shear_u_stamp_mpc)
    shear_v_stamp_mpc  = np.concatenate(all_shear_v_stamp_mpc)
    shear_g1_stamp     = np.concatenate(all_shear_g1_stamp)
    shear_g2_stamp     = np.concatenate(all_shear_g2_stamp)
    shear_g1_sc        = np.concatenate(all_g1_sc)
    shear_g2_sc        = np.concatenate(all_g2_sc)
    shear_g1_orig      = np.concatenate(all_shear_g1_orig)
    shear_g2_orig      = np.concatenate(all_shear_g2_orig)
    shear_z_stamp      = np.concatenate(all_shear_z_stamp)
    shear_ra_stamp_deg = np.concatenate(all_shear_ra_stamp_deg)
    shear_de_stamp_deg = np.concatenate(all_shear_de_stamp_deg)
    shear_bias_m_stamp = np.concatenate(all_shear_bias_m_stamp)
    shear_weight_stamp = np.concatenate(all_shear_weight_stamp)
    shear_sc = np.concatenate(all_shear_sc)



    logger.info('r_pair=%2.2fMpc    =%2.2farcmin ' , np.abs(halo1_u_rot_mpc - halo2_u_rot_mpc) , np.abs(halo1_u_rot_arcmin - halo2_u_rot_arcmin))
    logger.info('using %d galaxies for that pair' , len(shear_u_stamp_arcmin))

    grid_u_mpc = np.arange( -range_u_mpc / 2. , range_u_mpc / 2., config['pixel_size_mpc'] )
    grid_v_mpc = np.arange( -range_v_mpc / 2. , range_v_mpc / 2., config['pixel_size_mpc'] )

    logger.info('len grid_u_mpc=%d grid_v_mpc=%d' , len(grid_u_mpc) ,  len(grid_v_mpc))
    
    hist_g1, _, _    = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_g1_stamp * shear_weight_stamp)
    hist_g2, _, _    = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_g2_stamp * shear_weight_stamp)
    hist_g1sc, _, _  = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_g1_sc * shear_weight_stamp)
    hist_g2sc, _, _  = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_g2_sc * shear_weight_stamp)
    hist_n,  _, _    = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) )
    hist_scinv, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=1./shear_sc )
    hist_m , _, _    = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=(1+shear_bias_m_stamp) * shear_weight_stamp )
    hist_w , _, _    = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_weight_stamp)
    hist_w_sq , _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_weight_stamp**2)

    mean_g1 = hist_g1  / hist_m
    mean_g2 = hist_g2  / hist_m

    mean_scinv = hist_scinv / hist_m
    mean_g1sc = hist_g1sc / hist_m
    mean_g2sc = hist_g2sc / hist_m

    # in case we divided by zero
    mean_g1[hist_n == 0] = 0
    mean_g2[hist_n == 0] = 0
    mean_g1sc[hist_n == 0] = 0
    mean_g2sc[hist_n == 0] = 0
    hist_w[hist_n == 0] = 0
    hist_w_sq[hist_n == 0] = 0
    hist_m[hist_n == 0] = 0

    select = np.isclose(hist_m,0)
    mean_g1[select] = 0
    mean_g2[select] = 0
    mean_g1sc[select] = 0
    mean_g2sc[select] = 0
    hist_w[select] = 0
    hist_w_sq[select] = 0
    hist_m[select] = 0

    u_mid_mpc,v_mid_mpc = plotstools.get_bins_centers(grid_u_mpc) , plotstools.get_bins_centers(grid_v_mpc)
    grid_2d_u_mpc , grid_2d_v_mpc = np.meshgrid(u_mid_mpc,v_mid_mpc,indexing='ij')

    # dtype_shears_stacked = { 'names' : ['u_mpc','v_mpc','u_arcmin','v_arcmin','ra_deg','dec_deg','g1','g2','scinv','weight','z','n_gals'] , 'formats' : ['f8']*11 + ['i8']*1 }
    binned_u_mpc = grid_2d_u_mpc.flatten('F')
    binned_v_mpc = grid_2d_v_mpc.flatten('F')
    binned_g1 = mean_g1.flatten('F')
    binned_g2 = mean_g2.flatten('F')
    binned_g1sc = mean_g1sc.flatten('F')
    binned_g2sc = mean_g2sc.flatten('F')
    binned_n = hist_n.flatten('F')
    binned_scinv = hist_scinv.flatten('F')
    binned_w = hist_w.flatten('F')
    binned_w_sq = hist_w_sq.flatten('F')

    u_mpc = binned_u_mpc[:,None]
    v_mpc = binned_v_mpc[:,None]
    g1 = binned_g1[:,None]
    g2 = binned_g2[:,None]
    g1sc = binned_g1sc[:,None]
    g2sc = binned_g2sc[:,None]

    mean_scinv = binned_scinv[:,None]
    weight = binned_w[:,None]
    weight_sq = binned_w_sq[:,None]
    n_gals = binned_n[:,None] # set all rows to 1
    # scinv = scinv[:,None]
    # z = lenscat_stamp['z'][:,None]
    
    pl.figure()
    pl.subplot(1,2,1)
    pl.imshow(mean_g1,interpolation="nearest")
    pl.colorbar()
    pl.subplot(1,2,2)
    pl.imshow(mean_g2,interpolation="nearest")
    pl.colorbar()
    # pl.figure()
    # pl.subplot(1,2,1)
    # plot_pair(halo1_u_rot_mpc , halo1_v_rot_mpc , halo2_u_rot_mpc , halo2_v_rot_mpc , u_mpc, v_mpc, g1sc, g2sc , close=False,nuse = 1,quiver_scale=15)
    # pl.subplot(1,2,2)
    # plot_pair(halo1_u_rot_mpc , halo1_v_rot_mpc , halo2_u_rot_mpc , halo2_v_rot_mpc , shear_u_stamp_mpc, shear_v_stamp_mpc, shear_g1_stamp, shear_g2_stamp , close=False,nuse = 10,quiver_scale=2)
    pl.show()

    # pairs_shear = np.concatenate([u_mpc,v_mpc,u_arcmin,v_arcmin,g1,g2,mean_scinv,g1sc,g2sc,weight,weight_sq,n_gals],axis=1)
    # pairs_shear = tabletools.array2recarray(pairs_shear,dtype_shears_stacked)        


    # load pairs catalog
    # perform selection of pairs .. start with most massive
    # for pair in pairs
    #  load galaxies from CFHTLenS catalog
    #  rotate, scale-1, scale-2
    #  1) pixelise
    #  2) fit to unpixelised
    

def main():

    global log , config , args

    description = 'fil..stack'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-c', '--filename_config', type=str, action='store', help='filename of config file')
    parser.add_argument('-f', '--first', type=int, action='store', default=0, help='first item to process')
    parser.add_argument('-n', '--num', type=int, action='store', default=-1, help='number of items to process')
    parser.add_argument('-d', '--dir', type=str, action='store', default='./', help='directory to use')

    args = parser.parse_args()
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    logger.setLevel(logging_level)  
    config=yaml.load(open(args.filename_config))
    
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    # stack_halos() ; 
    # fit_stacked_halos()
    # plot_stack_fit()
    # stack_pairs()
    stack_pairs_new()

    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

main()