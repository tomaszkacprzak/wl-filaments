import sys, os, logging, yaml, argparse, time
import pylab as pl
import numpy as np
import tabletools, plotstools, mathstools, cosmology
import filaments_tools
import filaments_model_2hf
import filaments_model_1h

logging_level = logging.INFO
log = logging.getLogger("my_script") 
log.setLevel(logging_level)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s", "%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

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
            if 'star_flag' in cfhtlens_shear_catalog.dtype.names:
                select = cfhtlens_shear_catalog['star_flag'] == 0
                cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
                select = cfhtlens_shear_catalog['fitclass'] == 0
                cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
                log.info('removed stars, remaining %d' , len(cfhtlens_shear_catalog))

                select = (cfhtlens_shear_catalog['e1'] != 0.0) * (cfhtlens_shear_catalog['e2'] != 0.0)
                cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
                log.info('removed zeroed shapes, remaining %d' , len(cfhtlens_shear_catalog))


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

        log.info(ihalo)


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
    log.info('saved stacked_halos.png')
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
    log.info('using different sigma_g per pixel mean(inv_sq_sigma_g)=%2.5f len(inv_sq_sigma_g)=%d' , np.mean(fitobj.inv_sq_sigma_g) , len(fitobj.inv_sq_sigma_g))


    log.info('running grid search')
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
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    scaling = np.mean(pairs['R_pair'])/2.
    print scaling

    id_pair_first = args.first
    id_pair_last = args.first + args.num 

    vec_u_mpc, vec_v_mpc = np.arange(-2,2,0.05)*scaling, np.arange(-2,2,0.05)*scaling
    grid_u_mpc, grid_v_mpc = np.meshgrid(plotstools.get_bins_centers(vec_u_mpc) , plotstools.get_bins_centers(vec_v_mpc),indexing='ij')
    shear_g1 = np.zeros_like(grid_u_mpc)
    shear_g2 = np.zeros_like(grid_u_mpc) 
    shear_n  = np.zeros_like(grid_u_mpc) 

    halo1_u = scaling
    halo1_v = 0
    halo2_u = -scaling
    halo2_v = 0

    n_used=0
    list_z = []

    # pairs = tabletools.appendColumn(rec=pairs,arr=np.zeros(len(pairs)),name='area_arcmin2',dtype='f4')
    # pairs = tabletools.appendColumn(rec=pairs,arr=np.zeros(len(pairs)),name='n_eff',dtype='f4')
    for id_pair in range(id_pair_first,id_pair_last):
        
        if '.fits' in config['filename_shears']:
            shears_info = tabletools.loadTable(config['filename_shears'],hdu=id_pair+1)
        elif '.pp2' in config['filename_shears']:
            shears_info = tabletools.loadPickle(config['filename_shears'],pos=id_pair)

        n_gals_total=np.sum(shears_info['n_gals'])
        area=(shears_info['v_arcmin'].max() - shears_info['v_arcmin'].min())*(shears_info['u_arcmin'].max() - shears_info['u_arcmin'].min())
        pairs['n_gal'][id_pair] = n_gals_total
        pairs['area_arcmin2'][id_pair] = area
        pairs['n_eff'][id_pair] = float(n_gals_total)/area

      
        mean_mass=(halo1['m200'][id_pair] + halo2['m200'][id_pair])/2
        if mean_mass<13.5:
            continue
        print "==============================="
        print 'id_pair' , id_pair
        print 'pairs[ih1]'  , pairs['ih1'][id_pair]
        print 'pairs[ih2]'  , pairs['ih2'][id_pair]
        print 'halo1[m200]' , halo1['m200'][id_pair]
        print 'halo2[m200]' , halo2['m200'][id_pair]
        print 'total n_gals' , np.sum(shears_info['n_gals'])
        print "==============================="

        shears_info['u_mpc'] = shears_info['u_mpc'] / pairs['u1_mpc'][id_pair] * scaling
        shears_info['v_mpc'] = shears_info['v_mpc'] / pairs['u1_mpc'][id_pair] * scaling
        pairs_current = pairs[id_pair].copy()
        pairs_current['u1_mpc'] = pairs['u1_mpc'][id_pair] / pairs['u1_mpc'][id_pair] * scaling
        pairs_current['v1_mpc'] = pairs['v1_mpc'][id_pair] / pairs['u1_mpc'][id_pair] * scaling
        pairs_current['u2_mpc'] = pairs['u2_mpc'][id_pair] / pairs['u1_mpc'][id_pair] * scaling
        pairs_current['v2_mpc'] = pairs['v2_mpc'][id_pair] / pairs['u1_mpc'][id_pair] * scaling

        hist_g1, _, _    = np.histogram2d( x=shears_info['u_mpc'], y=shears_info['v_mpc'] , bins=(vec_u_mpc,vec_v_mpc) , weights=shears_info['g1']*shears_info['n_gals'])
        hist_g2, _, _    = np.histogram2d( x=shears_info['u_mpc'], y=shears_info['v_mpc'] , bins=(vec_u_mpc,vec_v_mpc) , weights=shears_info['g2']*shears_info['n_gals'])
        hist_n,  _, _    = np.histogram2d( x=shears_info['u_mpc'], y=shears_info['v_mpc'] , bins=(vec_u_mpc,vec_v_mpc) , weights=shears_info['n_gals'])

        hist_g1[np.isnan(hist_g1)]=0
        hist_g2[np.isnan(hist_g2)]=0
        shear_g1 += hist_g1
        shear_g2 += hist_g2
        shear_n += hist_n


        list_z.append(pairs[id_pair]['z'])
        n_used +=1
        log.info('stacked pair %d n_used %d' , id_pair , n_used)

    tabletools.saveTable(filename_pairs.replace('.fits','.fix.fits') ,pairs)

    mean_g1 = shear_g1  / shear_n
    mean_g2 = shear_g2  / shear_n
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
    fitobj.shear_w =  shears_info['weight']
    fitobj.set_shear_sigma()
    log.info('using different sigma_g per pixel mean(inv_sq_sigma_g)=%2.5f len(inv_sq_sigma_g)=%d' , np.mean(fitobj.inv_sq_sigma_g) , len(fitobj.inv_sq_sigma_g))

    
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
    
    fitobj.parameters[2]['box']['min'] = 11
    fitobj.parameters[2]['box']['max'] = 15
    fitobj.parameters[2]['n_grid'] = config['h1M200']['n_grid']
    
    fitobj.parameters[3]['box']['min'] = 11
    fitobj.parameters[3]['box']['max'] = 15
    fitobj.parameters[3]['n_grid'] = config['h2M200']['n_grid']

    filename_results_grid = 'stack.grid.pp2'
    filename_results_prob = 'stack.prob.pp2'

    if config['optimization_mode'] == 'gridsearch':
            log.info('running grid search')

            log_post , params, grids = fitobj.run_gridsearch()
            tabletools.savePickle(filename_results_prob,log_post.astype(np.float32),append=True)
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
            

    


def main():

    global log , config , args

    description = 'my_script'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-c', '--filename_config', type=str, action='store', help='filename of config file')
    parser.add_argument('-f', '--first', type=int, action='store', default=0, help='first item to process')
    parser.add_argument('-n', '--num', type=int, action='store', default=1, help='number of items to process')
    parser.add_argument('-d', '--dir', type=str, action='store', default='./', help='directory to use')

    args = parser.parse_args()
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    log.setLevel(logging_level)  
    config=yaml.load(open(args.filename_config))
    
    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    stack_pairs()
    # stack_halos() ; fit_stacked_halos()
    pl.show()

    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

main()