import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools, plotstools, nfw
import numpy as np
import pylab as pl
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import filaments_model_1h
import warnings
# warnings.simplefilter('ignore')

cospars = cosmology.cosmoparams()


def main():

    description = 'test_mod1h'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    # parser.add_argument('-c', '--filename_config', default='test2.yaml',type=str, action='store', help='name of the yaml config file')
    # parser.add_argument('-d', '--dry', default=False,  action='store_true', help='Dry run, dont generate data')

    args = parser.parse_args()
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    log = logging.getLogger("test_mod1h") 
    log.setLevel(logging_level)

    zclust= 0.257707923651
    M_200=4.7779e+14 
    concentr = 5.72/(1.+zclust)**0.71 * (M_200 / 1e14)**(-0.081)
    theta_cx=42.6863732955 
    zsource=1
    Omega_m=0.271
    theta_cx=15 
    theta_cy=0

    true_M200 = M_200

    x,y = np.meshgrid(np.linspace(-30,30,100),np.linspace(-30,30,100))
    shear_u_arcmin , shear_v_arcmin = x.flatten() , y.flatten()

    fitobj = filaments_model_1h.modelfit()
    fitobj.shear_z = zsource
    fitobj.shear_u_arcmin =  shear_u_arcmin
    fitobj.shear_v_arcmin =  shear_v_arcmin
    fitobj.halo_u_arcmin =  theta_cx
    fitobj.halo_v_arcmin =  theta_cy
    fitobj.halo_z =  zclust
    fitobj.sigma_g =  0.1
    fitobj.nh = nfw.NfwHalo()
    fitobj.nh.concentr = concentr
    fitobj.nh.z_cluster= fitobj.halo_z
    fitobj.nh.z_source = fitobj.shear_z
    fitobj.nh.theta_cx = fitobj.halo_u_arcmin
    fitobj.nh.theta_cy = fitobj.halo_v_arcmin 
    fitobj.nh.M_200 = M_200

    fitobj.shear_g1 , fitobj.shear_g2 , limit_mask , _ , _  =  fitobj.draw_model([true_M200])  
    fitobj.plot_shears(fitobj.shear_g1,fitobj.shear_g2,limit_mask,unit='arcmin',quiver_scale=1)


    print 'halo_u_arcmin' ,'halo_v_arcmin' ,fitobj.halo_u_arcmin, fitobj.halo_v_arcmin
    print 'concentr', concentr
    print 'm200', M_200
    print 'halo_z', fitobj.halo_z


    pl.show()

    data = np.concatenate( [fitobj.shear_g1[:,None] ,  fitobj.shear_g2[:,None] , fitobj.shear_u_arcmin[:,None] , fitobj.shear_v_arcmin[:,None] ], axis=1 )
    # np.savetxt('test_nfw_data.txt',data,header='g1 g2 u_arcmin v_arcmin (z_s=1)')
    np.savetxt('test_nfw_data.txt',data)
    log.info('saved test_nfw_data.txt')


    nh = nfw.NfwHalo()
    nh.M_200=M_200
    nh.concentr=concentr
    nh.z_cluster=fitobj.halo_z
    nh.theta_cx = fitobj.halo_u_arcmin
    nh.theta_cy = fitobj.halo_v_arcmin 
    
    theta_x=fitobj.shear_u_arcmin[:,None]
    theta_y=fitobj.shear_v_arcmin[:,None]

    print 'theta_x', theta_x[0]
    print 'theta_y', theta_y[0]

    [g1 , g2 , Delta_Sigma, Sigma_crit, kappa]=nh.get_shears(theta_x , theta_y, zsource)

    data = np.concatenate( [g1 ,  g2 , theta_x , theta_y], axis=1 )
    np.savetxt('test_nfw_data_gravlenspy.txt',data)
    log.info('saved test_nfw_data_gravlenspy.txt')


    print 'now run test_nfw_shears.m in matlab'

    # pl.savefig(filename_fig)
    # log.info('saved %s' % filename_fig)
    # pl.show()

    # pair_info = pairs_table[id_pair]

    # import pdb; pdb.set_trace()

    # log_post , grid_M200 = fitobj.run_gridsearch()
    # log_post = log_post - max(log_post)
    # norm = np.sum(np.exp(log_post))
    # prob_post = np.exp(log_post) 
    # pl.figure()
    # pl.plot(grid_M200 , log_post , '.-')
    # pl.figure()
    # pl.plot(grid_M200 , prob_post , '.-')
    # plotstools.adjust_limits()
    # pl.show()

    # fitobj.run_mcmc()
    # print fitobj.sampler
    # pl.figure()
    # pl.hist(fitobj.sampler.flatchain, bins=np.linspace(13,18,100), color="k", histtype="step")
    # # pl.plot(fitobj.sampler.flatchain,'x')
    # pl.show()
    # median_m = [np.median(fitobj.sampler.flatchain)]
    # print median_m
    # fitobj.plot_model(median_m)
    # filename_fig = 'halo_model_median.png'
    # pl.savefig(filename_fig)
    # log.info('saved %s' % filename_fig)


    # import pdb;pdb.set_trace()

    # # grid_search(pair_info,shears_info)
    # # test_model(shears_info,pair_info)


main()