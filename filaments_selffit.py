import os, yaml, argparse, sys, logging , pyfits,  emcee, tabletools, cosmology, filaments_tools, nfw, plotstools, filament
import numpy as np
import pylab as pl
import warnings
import filaments_model_2hf

warnings.simplefilter('once')

log = logging.getLogger("fil_selffit") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

redshift_offset = 0.2
weak_limit = 1000

def self_fit():


    fixed_kappa  = 0.0
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

    id_pair = 48

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
    # log.info('using sigma_g=%2.5f' , fitobj.sigma_g)

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
        # log.info('running mcmc search')
        # fitobj.n_walkers=10
        # fitobj.n_samples=2000
        # fitobj.run_mcmc()
        # params = fitobj.sampler.flatchain

        # plotstools.plot_dist(params)
        # pl.show()

    # vmax_post , best_model_g1, best_model_g2 , limit_mask,  vmax_params = fitobj.get_grid_max(log_post , params)


if __name__=='__main__':

    self_fit()