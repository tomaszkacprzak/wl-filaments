import os, yaml, argparse, sys, logging , pyfits,  emcee, tabletools, cosmology, filaments_tools, nfw, plotstools, filament
import numpy as np
import pylab as pl
import warnings
import filaments_model_2hf
import filaments_analyse

warnings.simplefilter('once')

logger = logging.getLogger("fil_selffit") 
logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False

nx = 25
ny = 25
grid_x = np.linspace(-12,12,nx)
grid_y = np.linspace(-12,12,ny)
filament_ds = 0.3
filament_radius = 1

def plot_overlap():

    filaments_analyse.config=config
    filaments_analyse.args=args
    X,Y = np.meshgrid(grid_x,grid_y,indexing='ij')
    D = np.zeros_like(X)
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')
    filename_grid = '%s/results.grid.%s.pp2' % (args.results_dir,name_data)
    grid_pickle = tabletools.loadPickle(filename_grid)

    n_result=0
    for ix,x_pos in enumerate(grid_x):
        for iy,y_pos in enumerate(grid_y):


            grid_kappa0 = grid_pickle['grid_kappa0'][:,:,0,0]
            grid_radius = grid_pickle['grid_radius'][:,:,0,0]
            filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir, n_result, n_result+1 , name_data)
            try:
                results_pickle = tabletools.loadPickle(filename_pickle,log=0)
            except:
                n_result+=1
                logger.info('missing %s' , filename_pickle)
                continue
            
            log_prob = results_pickle['log_post'][:,:,0,0]
            max0, max1 = np.unravel_index(log_prob.argmax(), log_prob.shape)
            D[ix,iy]=grid_kappa0[max0,max1]-filament_ds
            print max0, max1



            n_result+=1

    pl.figure()
    pl.pcolormesh(X,Y,D)
    pl.colorbar()
    pl.show()

    import pdb; pdb.set_trace()








def test_overlap():

    filename_pairs =  config['filename_pairs']                                   # pairs_bcc.fits'
    filename_halo1 =  config['filename_pairs'].replace('.fits' , '.halos1.fits') # pairs_bcc.halos1.fits'
    filename_halo2 =  config['filename_pairs'].replace('.fits' , '.halos2.fits') # pairs_bcc.halos2.fits'
    filename_shears = config['filename_shears']                                  # args.filename_shears 
    filename_shears_overlap = filename_shears.replace('.pp2','.overlap.pp2')
    filename_pairs_overlap = filename_pairs.replace('.fits','.overlap.fits')
    filename_halo1_overlap = filename_halo1.replace('.halos1.fits','.overlap.halos1.fits')
    filename_halo2_overlap = filename_halo2.replace('.halos2.fits','.overlap.halos2.fits')

    if os.path.isfile(filename_shears_overlap):
        os.remove(filename_shears_overlap)
        logger.warning('overwriting file %s' , filename_shears_overlap)

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    sigma_g_add =  0.0001

    id_pair = 3
    shears_info = tabletools.loadPickle(filename_shears,id_pair)

    overlapping_halo_m200 = 2 # x 1e14
    overlapping_halo_z = 0.3
    no_m200 = 1e-8

    n_pairs = len(grid_x) * len(grid_y)

    pairs_table_overlap = pairs_table[np.ones([n_pairs],dtype=np.int32)*id_pair]
    halo1_table_overlap = halo1_table[np.ones([n_pairs],dtype=np.int32)*id_pair]
    halo2_table_overlap = halo2_table[np.ones([n_pairs],dtype=np.int32)*id_pair]

    pairs_table_overlap['ipair'] = range(len(pairs_table_overlap))

    tabletools.saveTable(filename_pairs_overlap,pairs_table_overlap)
    tabletools.saveTable(filename_halo1_overlap,halo1_table_overlap)
    tabletools.saveTable(filename_halo2_overlap,halo2_table_overlap)

    for x_pos in grid_x:
        for y_pos in grid_y:

            logger.info('dx = %2.2f dy = %2.2f' % (x_pos,y_pos))

            fitobj = filaments_model_2hf.modelfit()
            fitobj.get_bcc_pz(config['filename_pz'])
            fitobj.kappa_is_K = False
            fitobj.R_start = config['R_start']
            fitobj.Dlos = pairs_table[id_pair]['Dlos']        
            fitobj.Dtot = np.sqrt(pairs_table[id_pair]['Dxy']**2+pairs_table[id_pair]['Dlos']**2)
            fitobj.boost = fitobj.Dtot/pairs_table[id_pair]['Dxy']
            fitobj.use_boost = config['use_boost']

            fitobj.shear_v_arcmin =  shears_info['v_arcmin']
            fitobj.shear_u_arcmin =  shears_info['u_arcmin']
            fitobj.shear_u_mpc =  shears_info['u_mpc']
            fitobj.shear_v_mpc =  shears_info['v_mpc']

            fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
            fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
            fitobj.halo1_u_mpc =  pairs_table['u1_mpc'][id_pair]
            fitobj.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair]
            fitobj.halo1_z =  pairs_table['z'][id_pair]

            fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
            fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
            fitobj.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair]
            fitobj.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair]
            fitobj.halo2_z =  pairs_table['z'][id_pair]

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
            shear_model_g1, shear_model_g2, limit_mask , _ , _  = fitobj.draw_model([filament_ds, filament_radius, no_m200, no_m200])
            fitobj.nh2.theta_cy = fitobj.halo2_v_arcmin
            fitobj.nh2.theta_cx = fitobj.halo2_u_arcmin

            # second fitobj ---------- overlapping halo
            fitobj2 = filaments_model_2hf.modelfit()
            fitobj2.get_bcc_pz(config['filename_pz'])
            fitobj2.use_boost = True
            fitobj2.kappa_is_K = False
            fitobj2.R_start = config['R_start']
            fitobj2.Dlos = pairs_table[id_pair]['Dlos']        
            fitobj2.Dtot = np.sqrt(pairs_table[id_pair]['Dxy']**2+pairs_table[id_pair]['Dlos']**2)
            fitobj2.boost = fitobj.Dtot/pairs_table[id_pair]['Dxy']
            fitobj2.use_boost = config['use_boost']

            fitobj2.shear_v_arcmin =  shears_info['v_arcmin']
            fitobj2.shear_u_arcmin =  shears_info['u_arcmin']
            fitobj2.shear_u_mpc =  shears_info['u_mpc']
            fitobj2.shear_v_mpc =  shears_info['v_mpc']

            fitobj2.halo1_z =  overlapping_halo_z
            fitobj2.halo1_u_arcmin = x_pos/cosmology.get_ang_diam_dist(overlapping_halo_z) * 180. / np.pi * 60.
            fitobj2.halo1_v_arcmin = y_pos/cosmology.get_ang_diam_dist(overlapping_halo_z) * 180. / np.pi * 60.
            fitobj2.halo1_u_mpc =  x_pos
            fitobj2.halo1_v_mpc =  y_pos

            fitobj2.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
            fitobj2.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
            fitobj2.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair]
            fitobj2.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair]
            fitobj2.halo2_z =  pairs_table['z'][id_pair]


            fitobj2.pair_z  = (fitobj2.halo1_z + fitobj2.halo2_z) / 2.

            fitobj2.filam = filament.filament()
            fitobj2.filam.pair_z =fitobj2.pair_z
            fitobj2.filam.grid_z_centers = fitobj2.grid_z_centers
            fitobj2.filam.prob_z = fitobj2.prob_z
            fitobj2.filam.set_mean_inv_sigma_crit(fitobj2.filam.grid_z_centers,fitobj2.filam.prob_z,fitobj2.filam.pair_z)

            fitobj2.nh1 = nfw.NfwHalo()
            fitobj2.nh1.z_cluster= fitobj2.halo1_z
            fitobj2.nh1.theta_cx = fitobj2.halo1_u_arcmin
            fitobj2.nh1.theta_cy = fitobj2.halo1_v_arcmin 
            fitobj2.nh1.set_mean_inv_sigma_crit(fitobj2.grid_z_centers,fitobj2.prob_z,fitobj2.pair_z)

            fitobj2.nh2 = nfw.NfwHalo()
            fitobj2.nh2.z_cluster= fitobj2.halo2_z
            fitobj2.nh2.theta_cx = fitobj2.halo2_u_arcmin
            fitobj2.nh2.theta_cy = fitobj2.halo2_v_arcmin 
            fitobj2.nh2.set_mean_inv_sigma_crit(fitobj2.grid_z_centers,fitobj2.prob_z,fitobj2.pair_z)

            shear_model_g1_neighbour, shear_model_g2_neighbour, limit_mask , _ , _  = fitobj2.draw_model([0.0, 0.5, overlapping_halo_m200, no_m200 ])

            do_plot=False
            if do_plot:

                cmax  = np.max([ np.abs(shear_model_g1_neighbour.min()),np.abs(shear_model_g1_neighbour.min()) , np.abs(shear_model_g1_neighbour.max()),np.abs(shear_model_g1_neighbour.max()),np.abs(shear_model_g1.max()),np.abs(shear_model_g1.max()),np.abs(shear_model_g1.min()),np.abs(shear_model_g1.min())])
                pl.figure(figsize=(20,10))
                pl.scatter( fitobj.shear_u_mpc  , fitobj.shear_v_mpc  , s=100, c=shear_model_g1+shear_model_g1_neighbour, lw=0)
                pl.clim(-cmax,cmax)
                pl.colorbar()
                pl.scatter( fitobj2.halo1_u_mpc , fitobj2.halo1_v_mpc , s=100, lw=0)
                pl.scatter( fitobj.halo1_u_mpc , fitobj.halo1_v_mpc , s=100)
                pl.scatter( fitobj.halo2_u_mpc , fitobj.halo2_v_mpc , s=100)
                pl.axis('equal')

                pl.show()

            fitobj.shear_g1 =  shear_model_g1 + np.random.randn(len(fitobj.shear_u_arcmin))*sigma_g_add
            fitobj.shear_g2 =  shear_model_g2 + np.random.randn(len(fitobj.shear_u_arcmin))*sigma_g_add
            fitobj.sigma_g =  np.std(shear_model_g2,ddof=1)
            fitobj.inv_sq_sigma_g = 1./sigma_g_add**2

            shears_info['g1'] = fitobj.shear_g1
            shears_info['g2'] = fitobj.shear_g2
            shears_info['weight'] = fitobj.inv_sq_sigma_g

            tabletools.savePickle(filename_shears_overlap,shears_info,append=True)          


def get_clone():

    mock_kappa0 = config['mock_kappa0']
    mock_radius = config['mock_radius']

    filename_pairs =  config['filename_pairs']                                   # pairs_bcc.fits'
    filename_halo1 =  config['filename_pairs'].replace('.fits' , '.halos1.fits') # pairs_bcc.halos1.fits'
    filename_halo2 =  config['filename_pairs'].replace('.fits' , '.halos2.fits') # pairs_bcc.halos2.fits'
    filename_shears = config['filename_shears']                                  # args.filename_shears 
    filename_shears_mock = config['filename_shears'].replace('.pp2','.clone.pp2')                                  # args.filename_shears 
    if os.path.isfile(filename_shears_mock): 
        os.remove(filename_shears_mock)
        logger.warning('overwriting file %s' , filename_shears_mock)


    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    # make clone using only those pairs, which were used in main analysis
    select = pairs_table['analysis'] == 1
    logger.info('using %d pairs that were selected for analysis' , sum(select))
    pairs_table_clone = pairs_table[select].copy()
    halo1_table_clone = halo1_table[select].copy()
    halo2_table_clone = halo2_table[select].copy()
    for irep in range(1,config['n_clones']):
        pairs_table_clone = np.concatenate([pairs_table_clone,pairs_table[select].copy()])
        halo1_table_clone = np.concatenate([halo1_table_clone,halo1_table[select].copy()])
        halo2_table_clone = np.concatenate([halo2_table_clone,halo2_table[select].copy()])
        logger.info('% 3d cloned % 3d pairs, total % 3d' %(irep,sum(select),len(pairs_table_clone)))

    pairs_table = pairs_table_clone
    halo1_table = halo1_table_clone
    halo2_table = halo2_table_clone

    id_pair_first = args.first
    id_pair_last = len(pairs_table_clone) if args.num==-1 else args.first + args.num
    if id_pair_first >= id_pair_last: raise Exception('first pair greater than len(halo_pairs) %d > %d' % (id_pair_first,id_pair_last))
    logger.info('running on pairs from %d to %d' , id_pair_first , id_pair_last)
  
    fitobj = filaments_model_2hf.modelfit()
    fitobj.get_bcc_pz(config['filename_pz'])
    prob_z = fitobj.prob_z
    grid_z_centers = fitobj.grid_z_centers
    grid_z_edges = fitobj.grid_z_edges

    for id_pair in range(id_pair_first,id_pair_last):

        id_shear = pairs_table[id_pair]['ipair']
        logger.info('--------- pair %d shear %d --------' , id_pair, id_shear) 
        # now we use that
        id_pair_in_catalog = id_pair
        if '.fits' in filename_shears:
            shears_info = tabletools.loadTable(filename_shears,hdu=id_shear+1)
        elif '.pp2' in filename_shears:
            shears_info = tabletools.loadPickle(filename_shears,pos=id_shear)
    
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

        shear_model_g1, shear_model_g2, limit_mask , _ , _  = fitobj.draw_model([mock_kappa0, mock_radius, mock_m200_h1, mock_m200_h2])

        if id_pair < 100:
            nuv = [pairs_table[id_pair]['nu'],pairs_table[id_pair]['nv']]
            u = np.reshape(shears_info['u_arcmin'],nuv,order='F')
            v = np.reshape(shears_info['v_arcmin'],nuv,order='F')
            g1 = np.reshape(shear_model_g1,nuv,order='F')
            g2 = np.reshape(shear_model_g2,nuv,order='F')
            pl.figure()
            pl.subplot(2,1,1)
            pl.pcolormesh(u,v,g1); 
            pl.axis('tight')
            pl.colorbar();
            pl.subplot(2,1,2)
            pl.pcolormesh(u,v,g2); 
            pl.axis('tight')
            pl.colorbar();
            pl.suptitle('%d m200_h1=%1.2e m200_h2=%1.2e' % (id_pair,mock_m200_h1,mock_m200_h2))
            filename_fig = 'figs/clone.shear.%05d.png' % id_pair
            pl.savefig(filename_fig)
            pl.close()
            logger.info('saved %s',filename_fig)

        sig = np.sqrt(1./ fitobj.inv_sq_sigma_g )
        select = np.isnan(sig) | np.isinf(sig)
        sig[select] = 0.
        noise_g1 = np.random.randn(len(shears_info['g1']))*sig
        noise_g2 = np.random.randn(len(shears_info['g2']))*sig

        neff = shears_info['weight']**2 /  shears_info['weight_sq'] 
        neff = neff[~select]

        fitobj.shear_g1 =  shear_model_g1 + noise_g1
        fitobj.shear_g2 =  shear_model_g2 + noise_g2

        shears_info['g1'] = fitobj.shear_g1
        shears_info['g2'] = fitobj.shear_g2

        tabletools.savePickle(filename_shears_mock,shears_info,append=True)

        logger.info('noise mean_n_gal=%2.2f mean_n_eff=%2.2f sig=%2.2f std_g1=%2.2f std_g2=%2.2f',np.mean(fitobj.shear_n_gals),np.mean(neff),np.mean(sig[~select]),np.std(noise_g1[~select]), np.std(noise_g2[~select]))

    pairs_table_clone['ipair']=np.arange(len(pairs_table_clone))
    tabletools.saveTable(filename_pairs.replace('.fits','.clone.fits'),pairs_table_clone)
    tabletools.saveTable(filename_halo1.replace('.halos1.fits','.clone.halos1.fits'),halo1_table_clone)
    tabletools.saveTable(filename_halo2.replace('.halos2.fits','.clone.halos2.fits'),halo2_table_clone)
    logger.warning('\n========================================\n \
                    now you have to change the config entry\n \
                    files to use clone files:\n \
                    filename_cfhtlens_shears : shears_cfhtlens_lrgs.clone.pp2\n \
                    filename_pairs : pairs_cfhtlens_lrgs.clone.fits\n \
                    ==========================================\
                    ')

def self_fit():


    fixed_kappa  = 0.05
    fixed_radius = 2
    fixed_m200 = 14
    fixed_m200 = 14

    filename_pairs = 'pairs_cfhtlens_null1.fits'
    filename_halo1 = 'pairs_cfhtlens_null1.halos1.fits'
    filename_halo2 = 'pairs_cfhtlens_null1.halos2.fits'
    filename_shears = 'shears_cfhtlens_g_null1.fits'
    filename_selffit = 'shears_selftest_kappa%2.2f.fits' % fixed_kappa

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    sigma_g_add =  0.
    fitobj = filaments_model_2hf.modelfit()
    pz = fitobj.get_bcc_pz('cfhtlens_cat_sample.fits')
    prob_z = fitobj.prob_z

    id_pair = 0

    shears_info = tabletools.loadTable(filename_shears,hdu=id_pair+1)
    fitobj = filaments_model_2hf.modelfit()
    fitobj.prob_z = prob_z

    fitobj.halo1_z = 0.2
    fitobj.halo2_z = 0.2
    fitobj.halo1_u_arcmin = 20
    fitobj.halo1_v_arcmin = 0
    fitobj.halo2_u_arcmin = -20
    fitobj.halo2_v_arcmin = 0
    fitobj.shear_v_arcmin =  shears_info['v_arcmin']
    fitobj.shear_u_mpc =  shears_info['u_mpc']
    fitobj.shear_v_mpc =  shears_info['v_mpc']

    fitobj.halo1_u_arcmin =  pairs_table['u1_arcmin'][id_pair]
    fitobj.halo1_v_arcmin =  pairs_table['v1_arcmin'][id_pair]
    fitobj.halo1_u_mpc =  pairs_table['u1_mpc'][id_pair]
    fitobj.halo1_v_mpc =  pairs_table['v1_mpc'][id_pair]
    fitobj.halo1_z =  pairs_table['z'][id_pair]

    fitobj.halo2_u_arcmin =  pairs_table['u2_arcmin'][id_pair]
    fitobj.halo2_v_arcmin =  pairs_table['v2_arcmin'][id_pair]
    fitobj.halo2_u_mpc =  pairs_table['u2_mpc'][id_pair]
    fitobj.halo2_v_mpc =  pairs_table['v2_mpc'][id_pair]
    fitobj.halo2_z =  pairs_table['z'][id_pair]

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

    fitobj.shear_u_arcmin =  shears_info['u_arcmin']
    fitobj.shear_v_arcmin =  shears_info['v_arcmin']

    shear_model_g1, shear_model_g2, limit_mask , _ , _ = fitobj.draw_model([fixed_kappa, fixed_radius, fixed_m200, fixed_m200])

    fitobj.shear_g1 =  shear_model_g1 + np.random.randn(len(shears_info['g1']))*sigma_g_add
    fitobj.shear_g2 =  shear_model_g2 + np.random.randn(len(shears_info['g2']))*sigma_g_add
    fitobj.sigma_g =  np.std(shear_model_g2,ddof=1)
    # fitobj.inv_sq_sigma_g = 1./sigma_g_add**2
    # logger.info('using sigma_g=%2.5f' , fitobj.sigma_g)

    fitobj.parameters[0]['box']['min'] = 0
    fitobj.parameters[0]['box']['max'] = 1
    fitobj.parameters[1]['box']['min'] = 1
    fitobj.parameters[1]['box']['max'] = 10
    fitobj.parameters[2]['box']['min'] = 14
    fitobj.parameters[2]['box']['max'] = 15
    fitobj.parameters[3]['box']['min'] = 14
    fitobj.parameters[3]['box']['max'] = 15

    # print 'halo1 m200' , halo1_table['m200'][id_pair]
    # print 'halo2 m200' , halo2_table['m200'][id_pair]

    shears_info['g1'] = fitobj.shear_g1
    shears_info['g2'] = fitobj.shear_g2
    fitobj.plot_shears(shears_info['g1'], shears_info['g2'],quiver_scale=0.1)
    pl.show()
    pl.scatter(shears_info['u_mpc'],shears_info['v_mpc'],c=np.abs(shears_info['g1'] + 1j*shears_info['g2'])); 
    pl.colorbar(); pl.show()

    
    tabletools.saveTable(filename_selffit, shears_info)
        
        # import pdb; pdb.set_trace()
        # # fitobj.plot_shears_mag(fitobj.shear_g1,fitobj.shear_g2)
        # # pl.show()
        # fitobj.save_all_models=False
        # logger.info('running mcmc search')
        # fitobj.n_walkers=10
        # fitobj.n_samples=2000
        # fitobj.run_mcmc()
        # params = fitobj.sampler.flatchain

        # plotstools.plot_dist(params)
        # pl.show()

    # vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_params = fitobj.get_grid_max(log_post , params)


def main():


    valid_actions = ['get_clone','test_overlap','plot_overlap']

    description = 'filaments_selffit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )
    parser.add_argument('-f', '--first', default=0,type=int, action='store', help='first pairs to process')
    parser.add_argument('-rd','--results_dir', action='store', help='where results files are' , default='results/' )
    parser.add_argument('-n', '--num', default=-1 ,type=int, action='store', help='last pairs to process')

    # parser.add_argument('-d', '--dry', default=False,  action='store_true', help='Dry run, dont generate data'), len(remove_list), len(set(remove_list)

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

    try:
        args.actions[0]
    except:
        raise Exception('choose one or more actions: %s' % str(valid_actions))

    for action in valid_actions:
        if action in args.actions:
            exec action+'()'
    for ac in args.actions:
        if ac not in valid_actions:
            print 'invalid action %s' % ac


    
if __name__=='__main__':
    main()
