import os, yaml, argparse, sys, logging , pyfits,  emcee, tabletools, cosmology, filaments_tools, nfw, plotstools, filament
import numpy as np
import pylab as pl
import warnings
import filaments_model_2hf

warnings.simplefilter('once')

logger = logging.getLogger("fil_selffit") 
logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False

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

    id_pair_first = args.first
    id_pair_last = len(pairs_table) if args.num==-1 else args.first + args.num
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


        mock_m200_h1 = pairs_table['m200_h1_fit'][id_pair_in_catalog]
        mock_m200_h2 = pairs_table['m200_h2_fit'][id_pair_in_catalog]

        shear_model_g1, shear_model_g2, limit_mask = fitobj.draw_model([mock_kappa0, mock_radius, mock_m200_h1, mock_m200_h2])

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

        sig = np.sqrt(1./shears_info['weight'])
        select = np.isnan(sig) | np.isinf(sig)
        sig[select] = 0.
        noise_g1 = np.random.randn(len(shears_info['g1']))*sig
        noise_g2 = np.random.randn(len(shears_info['g2']))*sig

        fitobj.shear_g1 =  shear_model_g1 + noise_g1
        fitobj.shear_g2 =  shear_model_g2 + noise_g2

        shears_info['g1'] = fitobj.shear_g1
        shears_info['g2'] = fitobj.shear_g2

        tabletools.savePickle(filename_shears_mock,shears_info,append=True)

        logger.info('noise sig=%2.2f std_g1=%2.2f std_g2=%2.2f',np.mean(sig[~select]),np.std(noise_g1[~select]), np.std(noise_g2[~select]))

    logger.warning('========================================\n \
                    now you have to change the config entry\n \
                    filename_cfhtlens_shears to use clone file:\n \
                    filename_cfhtlens_shears :shears_cfhtlens_lrgs.clone.pp2\n \
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

    shear_model_g1, shear_model_g2, limit_mask = fitobj.draw_model([fixed_kappa, fixed_radius, fixed_m200, fixed_m200])

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


    valid_actions = ['get_clone']

    description = 'filaments_selffit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )
    parser.add_argument('-f', '--first', default=0,type=int, action='store', help='first pairs to process')
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
