import os
import matplotlib as mpl
if 'DISPLAY' not in os.environ:
    mpl.use('tkagg')
import os, yaml, argparse, sys, logging , pyfits, emcee, tabletools, cosmology, filaments_tools, plotstools, mathstools, scipy, scipy.stats
import numpy as np
import matplotlib.pyplot as pl
print 'using matplotlib backend' , pl.get_backend()
# import matplotlib as mpl;
# from matplotlib import figure;
pl.rcParams['image.interpolation'] = 'nearest' ; 
import scipy.interpolate as interpl; import tktools as tt
from sklearn.neighbors import BallTree as BallTree
import cPickle as pickle
import filaments_model_1h
import filaments_model_1f
import filaments_model_2hf
import shutil
import warnings
warnings.simplefilter("once")


log = logging.getLogger("filam..fit") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

halos  = None
filename_halos_cfhtlens  = None
filename_cluscat  = None
filename_fields  = None
bossdr10  = None
pairs  = None
halo1  = None
halo2  = None
cluscat  = None
fieldscat  = None
cluscat_durret = None

def figure_density():
    
    global halos ,filename_halos_cfhtlens ,filename_cluscat ,filename_fields ,bossdr10 ,pairs ,halo1 ,halo2 ,cluscat ,fieldscat ,filename_cluscat_durret ,cluscat_durret

    pickle = tabletools.loadPickle(config['filename_pz'])
    prob_z =  pickle['prob_z']
    grid_z_centers = pickle['bins_z']
    grid_z_edges = plotstools.get_bins_edges(grid_z_centers)
    redshift_offset = 0.2

    import nfw
    nh1 = nfw.NfwHalo()
    nh1.z_cluster= 0.3
    nh1.M_200= 3.4e14
    nh1.concentr = nh1.get_concentr()
    nh1.set_mean_inv_sigma_crit(grid_z_centers,prob_z,nh1.z_cluster)
    nh1.R_200 = nh1.r_s*nh1.concentr      
    nh1.theta_cx = 0
    nh1.theta_cy = 0
    h1_r200_arcmin = nh1.R_200/cosmology.get_ang_diam_dist(nh1.z_cluster)/np.pi*180*60
    theta1_x=nh1.theta_cx+h1_r200_arcmin
    h1g1 , h1g2 , h1_DeltaSigma, Sigma_crit, kappa = nh1.get_shears_with_pz_fast(np.array([theta1_x,theta1_x,theta1_x]) , np.array([0,0,0]) , grid_z_centers , prob_z, redshift_offset)

    print h1g1 , h1g2  
    print h1_DeltaSigma 
    print Sigma_crit 
    print kappa
    print h1g1*Sigma_crit

    param0_old = 0.4
    param1 = 1.5
    param1_min = 0.75
    param1_max = 2.
    param0 = 0.5
    param0_min = 0.3
    param0_max = 7.5
    param0_prop = 1.5

    list_new = []
    list_old = []
    list_pro = []
    list_m200 = []
    list_radius = []
    list_min=[]
    list_max=[]
    list_r_min=[]
    list_r_max=[]


    for ip,vp in enumerate(pairs[pairs['analysis']==1]):

        if vp['analysis']!=1:
            continue
        
        nh1 = nfw.NfwHalo()
        nh1.z_cluster= halo1['z'][ip]
        nh1.M_200    = halo1['m200_fit'][ip]
        nh1.update()
        nh1.set_mean_inv_sigma_crit(grid_z_centers,prob_z,nh1.z_cluster)
        nh1.R_200 = nh1.r_s*nh1.concentr      
        nh1.theta_cx = vp['u1_arcmin']
        nh1.theta_cy = vp['v1_arcmin']
        h1_r200_arcmin = nh1.R_200/cosmology.get_ang_diam_dist(nh1.z_cluster)/np.pi*180*60
        theta1_x=nh1.theta_cx+h1_r200_arcmin
        h1g1 , h1g2 , h1_DeltaSigma, h1_Sigma_crit, h1_kappa = nh1.get_shears_with_pz_fast(np.array([theta1_x,theta1_x,theta1_x]) , np.array([0,0,0]) , grid_z_centers , prob_z, redshift_offset)
        h1g1m , h1g2m , h1_DeltaSigmam, h1_Sigma_critm, h1_kappam = nh1.get_shears_with_pz_fast(np.array([1e-5,1e-5,1e-5]) , np.array([0,0,0]) , grid_z_centers , prob_z, redshift_offset)

        nh2 = nfw.NfwHalo()
        nh2.z_cluster = halo2['z'][ip]
        nh2.M_200     = halo2['m200_fit'][ip]
        nh2.concentr = nh2.get_concentr()
        nh2.set_mean_inv_sigma_crit(grid_z_centers,prob_z,nh2.z_cluster)
        nh2.R_200 = nh2.r_s*nh2.concentr      
        nh2.theta_cx = vp['u2_arcmin']
        nh2.theta_cy = vp['v2_arcmin']
        h1_r200_arcmin = nh2.R_200/cosmology.get_ang_diam_dist(nh2.z_cluster)/np.pi*180*60
        theta1_x=nh2.theta_cx+h1_r200_arcmin
        h2g1 , h2g2 , h2_DeltaSigma, h2_Sigma_crit, h2_kappa = nh2.get_shears_with_pz_fast(np.array([theta1_x,theta1_x,theta1_x]) , np.array([0,0,0]) , grid_z_centers , prob_z, redshift_offset)
        h2g1m , h2g2m , h2_DeltaSigmam, h2_Sigma_critm, h2_kappam = nh2.get_shears_with_pz_fast(np.array([1e-5,1e-5,1e-5]) , np.array([0,0,0]) , grid_z_centers , prob_z, redshift_offset)
        
        DeltaSigma_at_R200 = (np.abs(h1_DeltaSigma[0])+np.abs(h2_DeltaSigma[0]))/2.
        filament_kappa0 = param0 * DeltaSigma_at_R200 / 1e14
        filament_radius = param1 
        filament_kappa0_min = param0_min * DeltaSigma_at_R200 / 1e14
        filament_kappa0_max = param0_max * DeltaSigma_at_R200 / 1e14
        filament_radius_min = param1_min 
        filament_radius_max = param1_max 

        DeltaSigma_at_R200_bug = (np.abs(h2g1[0]*h2_Sigma_crit)+np.abs(h1g1[0]*h1_Sigma_crit))/2.
        filament_kappa0_old = param0_old * DeltaSigma_at_R200_bug / 1e14

        filament_kappa0_prop = param0_prop * (-1)*(h1g1m[0]+h2g1m[0]) * h2_Sigma_critm / 1e14

        radius_grid = np.linspace(-3,3,20000)
        dr = radius_grid[1]-radius_grid[0]
        Dtot = np.sqrt(vp['Dxy']**2+vp['Dlos']**2)
        filament_mass_mid = np.sum(param0     * DeltaSigma_at_R200 / 1e14 * dr / (1+ radius_grid**2/(param1)**2) )* (Dtot-3) * 1e14
        filament_mass_min = np.sum(param0_min * DeltaSigma_at_R200 / 1e14 * dr / (1+ radius_grid**2/(param1_min)**2) )* (Dtot-3) * 1e14
        filament_mass_max = np.sum(param0_max * DeltaSigma_at_R200 / 1e14 * dr / (1+ radius_grid**2/(param1_max)**2) )* (Dtot-3) * 1e14

        print '% 4d M1=%2.2e M2=%2.2e Mmid=%2.2e DS200=%2.2e kappa0=%2.3f kappa0_bug=%2.3f radius=%2.2f Dtot=%2.2f' % (ip,nh1.M_200,nh2.M_200, (nh1.M_200+nh2.M_200 )/2. ,DeltaSigma_at_R200,filament_kappa0,filament_kappa0_old,filament_radius, Dtot)
        print '---- DSmid=%2.2e kappa0=%2.2f' % ((h1_DeltaSigmam[0]+h2_DeltaSigmam[0]),filament_kappa0)
        print '---- %5.3f %5.3e %5.3e %5.3f' % (h1g1[0], h1_DeltaSigma[0], h1_Sigma_crit, h1_kappa[0])
        print '---- %5.3f %5.3e %5.3e %5.3f' % (h2g1[0], h2_DeltaSigma[0], h2_Sigma_crit, h2_kappa[0])
        print '---- change: %2.2f' % (filament_kappa0_old/filament_kappa0)
        print '---- mass: %2.2e +/- %2.2e %2.2e' % (filament_mass_mid,filament_mass_max,filament_mass_min)
        print '---- kappa0: %2.2e +/- %2.2e %2.2e' % (filament_kappa0,filament_kappa0_max,filament_kappa0_min)



        list_m200.append((nh2.M_200+nh1.M_200)/2.)
        list_old.append(filament_kappa0_old)        
        list_new.append(filament_kappa0)
        list_min.append(filament_kappa0_min)
        list_max.append(filament_kappa0_max)
        list_pro.append(filament_kappa0_prop)        
        list_radius.append(filament_radius)
        list_r_min.append(filament_radius_min)
        list_r_max.append(filament_radius_max)

    list_m200 = np.array(list_m200)
    list_new = np.array(list_new)
    import fitting
    a,b,Cab = fitting.get_line_fit(list_m200/1e14,list_new,np.ones_like(list_new))
    amin,bmin,Cab = fitting.get_line_fit(list_m200/1e14,list_min,np.ones_like(list_new))
    amax,bmax,Cab = fitting.get_line_fit(list_m200/1e14,list_max,np.ones_like(list_new))
    print 'line fit DS a' ,  a
    print 'line fit DS b' ,  b
    print 'line fit DS C' ,  Cab
    print 'median Delta Sigma' , np.median(list_new)

    ar,br,Cabr = fitting.get_line_fit(list_m200/1e14,list_radius,np.ones_like(list_new))
    armin,brmin,Cabr = fitting.get_line_fit(list_m200/1e14,list_r_min,np.ones_like(list_new))
    armax,brmax,Cabr = fitting.get_line_fit(list_m200/1e14,list_r_max,np.ones_like(list_new))
    print 'line fit R a' ,  ar
    print 'line fit R b' ,  br
    print 'line fit R C' ,  Cabr
    print 'median Delta Sigma' , np.median(list_new)


    # pl.plot(list_m200,list_old,'r.')
    pl.figure()
    pl.plot(list_m200/1e14,list_new,'g.')
    pl.plot(list_m200/1e14,b*list_m200/1e14+a)
    pl.plot(list_m200/1e14,bmin*list_m200/1e14+amin)
    pl.plot(list_m200/1e14,bmax*list_m200/1e14+amax)
    pl.xlabel('mean m200')
    pl.ylabel(r'mean \Delta\Sigma')
    # pl.plot(list_m200,list_pro,'b.')

    pl.figure()
    pl.plot(list_m200/1e14,list_radius,'g.')
    pl.plot(list_m200/1e14,br*list_m200/1e14+ar)
    pl.plot(list_m200/1e14,brmin*list_m200/1e14+armin)
    pl.plot(list_m200/1e14,brmax*list_m200/1e14+armax)
    pl.xlabel('mean m200')
    pl.ylabel(r'mean radius')
    # pl.plot(list_m200,list_pro,'b.')
    pl.show()

    import pdb; pdb.set_trace()



def figures_individual():

    import filaments_analyse
    filaments_analyse.config=config
    args.results_dir='results'
    filaments_analyse.args=args

    
    global halos ,filename_halos_cfhtlens ,filename_cluscat ,filename_fields ,bossdr10 ,pairs ,halo1 ,halo2 ,cluscat ,fieldscat ,filename_cluscat_durret ,cluscat_durret

    n_pairs = len(halo1)
    print 'n_pairs' , n_pairs

    thres = 1.
    print 'BF n_clean '        ,   sum( (pairs['manual_remove']==0) & (pairs['BF1']>thres) & (pairs['BF2']>thres) & ( (pairs['eyeball_class']==1) | (pairs['eyeball_class']==3) )  )
    print 'BF n_contaminating ',   sum( (pairs['manual_remove']==0) & (pairs['BF1']>thres) & (pairs['BF2']>thres) & ( (pairs['eyeball_class']==2) | (pairs['eyeball_class']==0) )  )

    
    select = [233,152,47]

    for ids in select:
        
        prod_pdf, grid_dict, list_ids_used , n_pairs_used = filaments_analyse.get_prob_prod_gridsearch_2D([ids])

        print 'used %d pairs' % n_pairs_used

        for ic in list_ids_used:
            print 'ipair=%d ih1=%d ih2=%d m200_h1=%2.2f m200_h2=%2.2f' % (ic, pairs[ic]['ih1'] , pairs[ic]['ih2'], pairs[ic]['m200_h1_fit'], pairs[ic]['m200_h2_fit'])

        res_dict = { 'prob' : prod_pdf , 'params' : grid_dict, 'n_obj' : n_pairs_used }

        # single 2d plot

        max0, max1 = np.unravel_index(prod_pdf.argmax(), prod_pdf.shape)
        print grid_dict['grid_kappa0'][max0,max1], grid_dict['grid_radius'][max0,max1]
        max_radius = grid_dict['grid_radius'][0,max1]
        contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod_pdf,list_sigmas=[1,2,3,4,5])
        xlabel=r'$\Delta\Sigma$  $10^{14} \mathrm{M}_{\odot} \mathrm{Mpc}^{-2} h$'
        ylabel=r'radius $\mathrm{Mpc}/h$'
        
        pl.figure(1)
        cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='y')
        pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf)
        pl.axhline(max_radius,color='r')
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)
        pl.title('CFHTLens + BOSS-DR10, pair %d' % ids)
        pl.axis('tight')

        # paper plots in ugly colormap
        pl.figure(2)
        cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels[::2],colors='b')
        fmt = {}; strs = [ r'$1\sigma$', r'$3\sigma$', r'$5\sigma$'] ; 
        for l,s in zip( cp.levels, strs ): 
            fmt[l] = s
        pl.clabel(cp, cp.levels, fmt=fmt , fontsize=20)
        # pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,cmap=pl.cm.YlOrRd)
        cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],1], alpha=0.2 ,  colors=['b'] )

        # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[1],1], alpha=0.2 ,  colors=['b'])
        cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[2],1], alpha=0.2 ,  colors=['b'])
        # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[3],1], alpha=0.2 ,  colors=['b'])
        cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[4],1], alpha=0.2 ,  colors=['b'])
        # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],contour_levels[2]], alpha=0.25 ,  cmap=pl.cm.Blues)
        # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels, alpha=0.25 ,  cmap=pl.cm.bone)
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)
        pl.axis('tight')
        # pl.ylim([0,4])

        # plot 1d - just kappa0
        prob_kappa0 = np.sum(prod_pdf,axis=1)
        prob_radius = np.sum(prod_pdf,axis=0)
        grid_kappa0 = grid_dict['grid_kappa0'][:,0] 
        grid_radius = grid_dict['grid_radius'][0,:] 
        max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_kappa0,prob_kappa0)
        print 'kappa0 %2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

        pl.figure(2)
        pl.title('CFHTLens + BOSS-DR10, pair %d, sigma=%2.2f' % (ids,max_par/err_lo))

        max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_radius,prob_radius)
        print 'radius %2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)



        id_radius=max1
        at_radius=grid_dict['grid_radius'][0,id_radius]
        print 'using radius=' , at_radius
        kappa_at_radius=prod_pdf[:,id_radius].copy()
        kappa_at_radius/=np.sum(kappa_at_radius)
        kappa_grid = grid_dict['grid_kappa0'][:,id_radius]
        max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(kappa_grid,kappa_at_radius)
        print '%2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

        pl.figure(3)
        pl.plot(kappa_grid,kappa_at_radius)
        # pl.title('CFHTLens + BOSS-DR10, radius=%2.2f, using %d pairs' % (at_radius,n_pairs_used))
        pl.xlabel(xlabel)


        pl.show()



def figure_model():

    filename_pairs =  config['filename_pairs']                                   # pairs_bcc.fits'
    filename_halo1 =  config['filename_pairs'].replace('.fits' , '.halos1.fits') # pairs_bcc.halos1.fits'
    filename_halo2 =  config['filename_pairs'].replace('.fits' , '.halos2.fits') # pairs_bcc.halos2.fits'
    filename_shears = config['filename_shears']                                  # args.filename_shears 

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    sigma_g_add =  0.

    id_pair = 216
    shears_info = tabletools.loadPickle(filename_shears,id_pair)

    import filaments_model_2hf
    import filament
    import nfw
    fitobj = filaments_model_2hf.modelfit()
    fitobj.kappa_is_K = config['kappa_is_K']
    fitobj.get_bcc_pz(config['filename_pz'])

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

    fitobj.shear_u_arcmin =  shears_info['u_arcmin']

    fitobj.R_start = 1
    fitobj.Dlos = pairs_table[id_pair]['Dlos']        
    fitobj.Dtot = np.sqrt(pairs_table[id_pair]['Dxy']**2+pairs_table[id_pair]['Dlos']**2)
    fitobj.boost = fitobj.Dtot/pairs_table[id_pair]['Dxy']
    fitobj.use_boost = config['use_boost']

    param_radius = 1.5
    param_kappa0 = 0.5
    param_masses = 3
    shear_model_g1 , shear_model_g2 , limit_mask , model_DeltaSigma, model_kappa = fitobj.draw_model([param_kappa0, param_radius, param_masses, param_masses])


    nx = sum(np.isclose(fitobj.shear_u_mpc,fitobj.shear_u_mpc[0]))
    ny = sum(np.isclose(fitobj.shear_v_mpc,fitobj.shear_v_mpc[0]))
    print nx,ny
    shear_u_mpc = np.reshape(fitobj.shear_u_mpc,[nx,ny])
    shear_v_mpc = np.reshape(fitobj.shear_v_mpc,[nx,ny])
    shear_model_g1= np.reshape(shear_model_g1,[nx,ny])
    shear_model_g2= np.reshape(shear_model_g2,[nx,ny])
    limit_mask=np.reshape(limit_mask,[nx,ny])
    model_kappa=np.reshape(model_kappa,[nx,ny])
    model_DeltaSigma=np.reshape(model_DeltaSigma,[nx,ny])

  
    # pl.figure()
    # pl.pcolormesh( shear_u_mpc , shear_v_mpc , shear_model_g1)   
        
    # pl.figure()
    # pl.pcolormesh( shear_u_mpc , shear_v_mpc , shear_model_g2)
    
    # pl.figure(figsize=(27.3/3.,10/3.))
    pl.figure(figsize=(28/3.,10/3.))
    pl.subplots_adjust(bottom=0.15)
    n_levels=8

    # cmap = pl.get_cmap('PuBu')
    # pcm = pl.pcolormesh(shear_u_mpc,shear_v_mpc,model_DeltaSigma,cmap=cmap,norm=pl.matplotlib.colors.LogNorm())
    # pcm = pl.contourf(shear_u_mpc,shear_v_mpc,model_DeltaSigma,levels=[0.5e13,1e13,2e13,1e15,2e15],cmap=cmap,norm=pl.matplotlib.colors.LogNorm())
    cmap = pl.get_cmap('Greys')
    pcm = pl.contourf(shear_u_mpc,shear_v_mpc,np.log10(model_DeltaSigma),levels=np.linspace(12.,13.7,n_levels),cmap=cmap,extend='both')
    # pcm.cmap.set_over('black')
    # pcm.cmap.set_under('black')
    cmap = pl.get_cmap('Blues')
    pcm = pl.contourf(shear_u_mpc,shear_v_mpc,model_DeltaSigma,levels=np.logspace(12.,13.7,n_levels),cmap=cmap,norm=pl.matplotlib.colors.LogNorm())
    pl.contour(shear_u_mpc,shear_v_mpc,model_DeltaSigma,levels=np.logspace(12.,13.7,n_levels),colors='k',lw=2,zorder=1)
    # pcm = pl.pcolormesh(shear_u_mpc,shear_v_mpc,model_kappa,cmap=cmap,norm=pl.matplotlib.colors.LogNorm())
    # pcm = pl.pcolormesh(shear_u_mpc,shear_v_mpc,model_kappa,cmap=cmap)

    # import pdb; pdb.set_trace()
    ephi=0.5*np.arctan2(shear_model_g2,shear_model_g1)              

    if limit_mask==None:
        emag=np.sqrt(shear_model_g1**2+shear_model_g2**2)
    else:
        emag=np.sqrt(shear_model_g1**2+shear_model_g2**2) * limit_mask

    emag = np.reshape(emag,[nx,ny])
    ephi = np.reshape(ephi,[nx,ny])
    unit='Mpc'

    line_width=0.003
    quiver_scale = 0.7
    nuse = 4
   
    shear_u_mpc = shear_u_mpc[::nuse,::nuse]
    shear_v_mpc = shear_v_mpc[::nuse,::nuse]
    shear_model_g1 = shear_model_g1[::nuse,::nuse]
    shear_model_g2 = shear_model_g2[::nuse,::nuse]
    emag = emag[::nuse,::nuse]
    ephi = ephi[::nuse,::nuse]


    hc = 8.42
    rad = 1.5 # mpc
    mask = (np.sqrt((shear_u_mpc-hc)**2 + shear_v_mpc**2) >  (rad + 0.2)) * (np.sqrt((shear_u_mpc+hc)**2 + shear_v_mpc**2) > (rad + 0.2)) 
    shear_u_mpc = shear_u_mpc[mask]
    shear_v_mpc = shear_v_mpc[mask]
    shear_model_g1 = shear_model_g1[mask]
    shear_model_g2 = shear_model_g2[mask]
    emag = emag[mask]
    ephi = ephi[mask]
    

    pos=hc*1.1-np.pi/4.
    #pl.plot([-pos,pos],[ param_radius, param_radius],c='k')
    #pl.plot([-pos,pos],[-param_radius,-param_radius],c='k')

    import matplotlib
    nz = pl.matplotlib.colors.Normalize()
    nz.autoscale(emag)
    quiv=pl.quiver(shear_u_mpc,shear_v_mpc,emag*np.cos(ephi),emag*np.sin(ephi),linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',label='original',scale=quiver_scale , width = line_width)  
    # quiv=pl.quiver(shear_u_mpc,shear_v_mpc,emag*np.cos(ephi),emag*np.sin(ephi),linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',label='original',scale=quiver_scale , width = line_width, color=pl.matplotlib.cm.jet(nz(emag)))  
    # cax,_ = pl.matplotlib.colorbar.make_axes(pl.gca())
    # cb = pl.matplotlib.colorbar.ColorbarBase(cax, cmap=pl.matplotlib.cm.jet, norm=nz)
    # pl.colorbar(quiv)

    # pl.gca().add_patch(matplotlib.patches.Rectangle((5.5,2.3),5,10,color='w'))
    # qk = pl.quiverkey(quiv, 0.72, 0.8, 0.005, r'$g=0.005$', labelpos='E', coordinates='figure', fontproperties={'weight': 'bold' , 'size':20})
    pl.gca().add_patch(matplotlib.patches.Rectangle((7.5,-3.7),4.5,1.1,facecolor='white', edgecolor='black',zorder=2))
    #pl.gca().add_patch(matplotlib.patches.Circle((-hc,0),rad,edgecolor='k',facecolor='none',lw=1))
    #pl.gca().add_patch(matplotlib.patches.Circle(( hc,0),rad,edgecolor='k',facecolor='none',lw=1))
    qk = pl.quiverkey(quiv, 0.775, 0.22, 0.005, r'$g=0.005$', labelpos='E', coordinates='figure', fontproperties={'weight': 'bold' , 'size':16},zorder=3)
    qk.set_zorder(3)

    pl.xlabel(unit)
    pl.ylabel(unit)

    pl.xlim([min(shear_u_mpc.flatten())-0.5,max(shear_u_mpc.flatten())+0.5])
    pl.ylim([min(shear_v_mpc.flatten())-0.5,max(shear_v_mpc.flatten())+0.5])
    pl.axis('tight')
    pl.gca().tick_params('both', length=0, width=2, which='major')

    # cbaxes = pl.gcf().add_axes([0.4, 0.8, 0.23, 0.05]) 
    # cbar = pl.colorbar(pcm,cax=cbaxes, orientation='horizontal',ticks=[1e13,1e14,1e15])
    
    cbaxes = pl.gcf().add_axes([0.91, 0.2, 0.02, 0.63]) 
    # cbar = pl.colorbar(pcm,cax=cbaxes, orientation='horizontal',ticks=[1e13,1e14,1e15])
    # cbar = pl.colorbar(pcm,cax=cbaxes, ticks=[1e13,1e14,1e15])
    cbar = pl.colorbar(pcm,cax=cbaxes)
    cbar.set_ticks( np.append(np.linspace(1e12,1e13,10),np.linspace(1e13,1e14,10)) )
    # cbar.set_clim([model_DeltaSigma.flatten().min(),2.25e13])
    # cbar = pl.colorbar(pcm,cax=cbaxes)
    pl.figtext(0.935,0.8,r'$\Delta\Sigma$')

    print 'fitobj.nh2.R_200' , fitobj.nh2.R_200

    # print fitobj.nh2.r_200
    pl.show()

    import pdb; pdb.set_trace()

def figure_fields():

    global halos ,filename_halos_cfhtlens ,filename_cluscat ,filename_fields ,bossdr10 ,pairs ,halo1 ,halo2 ,cluscat ,fieldscat ,filename_cluscat_durret ,cluscat_durret

    box_w1 = [29.5,39.5,-12,-3]
    box_w2 = [208,221,50.5,58.5]
    box_w3 = [329.5,336,-2,5.5]
    box_w4 = [131.5,137.5,-6.5,-0.5]

    pairs_all = pairs.copy()
    pairs_all = pairs_all[pairs_all['Dxy']<30]
    select = pairs['analysis']==0
    pairs=pairs[select]
    halo1=halo1[select]
    halo2=halo2[select]
    n_pairs_used = len(pairs)

    print 'using %d pairs' % len(halo2)


    try:
        import matplotlib
        import matplotlib.gridspec as gridspec
    except:
        log.error('gridspec not found - no plot today')
        return None

    import matplotlib.gridspec as gridspec
    fig = pl.figure(figsize=(18,6))
    fig.clf()
    fig.subplots_adjust(left=0.08,right=0.875)
    # gs = gridspec.GridSpec( 2, 2, width_ratios=[1,1], height_ratios=[2,1], wspace=0.25, hspace=0.25)
    gs = gridspec.GridSpec( 1, 3 , wspace=0.15, hspace=0.1)
    # ax1 = fig.add_subplot(gs[0]) # 7x10
    # ax2 = fig.add_subplot(gs[1]) # 7x12
    # ax3 = fig.add_subplot(gs[2]) # 7x6
    # ax4 = fig.add_subplot(gs[3]) # 2x5

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    fig.text(0.5, 0.04, 'RA', ha='center', va='center' , fontsize=24)
    fig.text(0.03, 0.5, 'Dec', ha='center', va='center', rotation='vertical', fontsize=24)

    minz=0.2
    maxz=1
    halo_size = 100

    # ax1.scatter(cluscat_durret['ra'],cluscat_durret['de'],s=50,marker='d', vmin=minz, vmax=maxz)
    # ax1.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s', vmin=minz, vmax=maxz)
    ax1.text(box_w1[1]-0.2,box_w1[2]+0.2,'W1',fontsize=20)
    for i in range(len(pairs_all)): 
        ax1.plot([pairs_all['ra1'][i],pairs_all['ra2'][i]],[pairs_all['dec1'][i],pairs_all['dec2'][i]],c='#989898' ,lw=4,zorder=0)
        
    for i in range(len(pairs)): 
        ax1.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c='r',lw=4,zorder=1)
        # ax1.text(pairs['ra1'][i],pairs['dec1'][i],'%d'%pairs['ih1'][i],fontsize=10)
        # ax1.text(pairs['ra2'][i],pairs['dec2'][i],'%d'%pairs['ih2'][i],fontsize=10)
        # ax1.text((pairs['ra1'][i]+pairs['ra2'][i])/2.,(pairs['dec1'][i]+pairs['dec2'][i])/2.,'%d'%pairs[i]['ipair'],fontsize=10)
    ax1.scatter(pairs['ra1'],pairs['dec1'], halo_size , c=halo1['z'] , marker = 'o' , vmin=minz, vmax=maxz , zorder=2) #
    ax1.scatter(pairs['ra2'],pairs['dec2'], halo_size , c=halo2['z'] , marker = 'o' , vmin=minz, vmax=maxz , zorder=2) #
    for f in range(len(fieldscat)):
        x1=fieldscat[f]['ra_min']
        x2=fieldscat[f]['de_min']
        l1=fieldscat[f]['ra_max'] - fieldscat[f]['ra_min']
        l2=fieldscat[f]['de_max'] - fieldscat[f]['de_min']
        rect = matplotlib.patches.Rectangle((x1,x2), l1, l2, facecolor=None, edgecolor='Black', linewidth=1.0 , fill=None)
        ax1.add_patch(rect)

    ax1.set_xlim(box_w1[0],box_w1[1])
    ax1.set_ylim(box_w1[2],box_w1[3])
    ax1.invert_xaxis()
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.set_xticks([30,34,38])
    ax1.set_yticks([-4,-7,-10])
    ax1.tick_params(axis='both', which='major', labelsize=22)

    # ax2.scatter(cluscat_durret['ra'],cluscat_durret['de'],s=50,marker='d',vmin=minz, vmax=maxz)
    # ax2.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s',vmin=minz, vmax=maxz)
    ax2.text(box_w2[1]-0.2,box_w2[2]+0.2,'W3',fontsize=20)
    for i in range(len(pairs_all)): 
        ax2.plot([pairs_all['ra1'][i],pairs_all['ra2'][i]],[pairs_all['dec1'][i],pairs_all['dec2'][i]],c='#989898' ,lw=4,zorder=0)
    
    for i in range(len(pairs)): 
        ax2.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c='r',lw=4,zorder=1)
        # ax2.text(pairs['ra1'][i],pairs['dec1'][i],'%d'%pairs['ih1'][i],fontsize=10)
        # ax2.text(pairs['ra2'][i],pairs['dec2'][i],'%d'%pairs['ih2'][i],fontsize=10)
        # ax2.text((pairs['ra1'][i]+pairs['ra2'][i])/2.,(pairs['dec1'][i]+pairs['dec2'][i])/2.,'%d'%pairs[i]['ipair'],fontsize=10)
    ax2.scatter(pairs['ra1'],pairs['dec1'], halo_size , c=halo1['z'] , marker = 'o' ,vmin=minz, vmax=maxz,zorder=2) #
    ax2.scatter(pairs['ra2'],pairs['dec2'], halo_size , c=halo2['z'] , marker = 'o' ,vmin=minz, vmax=maxz,zorder=2) #      
    for f in range(len(fieldscat)):
        x1=fieldscat[f]['ra_min']
        x2=fieldscat[f]['de_min']
        l1=fieldscat[f]['ra_max'] - fieldscat[f]['ra_min']
        l2=fieldscat[f]['de_max'] - fieldscat[f]['de_min']
        rect = matplotlib.patches.Rectangle((x1,x2), l1, l2, facecolor=None, edgecolor='Black', linewidth=1.0 , fill=None)
        ax2.add_patch(rect)


    ax2.set_xlim(box_w2[0],box_w2[1])
    ax2.set_ylim(box_w2[2],box_w2[3])
    ax2.invert_xaxis()
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.set_xticks([209,214,219])
    ax2.set_yticks([52,55,58])
    ax2.tick_params(axis='both', which='major', labelsize=22)

    ax3.text(box_w3[1]-0.2,box_w3[2]+0.2,'W4',fontsize=20)
    # ax3.scatter(cluscat_durret['ra'],cluscat_durret['de'],s=50,marker='d',vmin=minz, vmax=maxz)
    # ax3.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s',vmin=minz, vmax=maxz)
    for i in range(len(pairs_all)): 
        ax3.plot([pairs_all['ra1'][i],pairs_all['ra2'][i]],[pairs_all['dec1'][i],pairs_all['dec2'][i]],c='#989898' ,lw=4,zorder=0)
    
    for i in range(len(pairs)): 
        ax3.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c='r',lw=4,zorder=1)
        # ax3.text(pairs['ra1'][i],pairs['dec1'][i],'%d'%pairs['ih1'][i],fontsize=10)
        # ax3.text(pairs['ra2'][i],pairs['dec2'][i],'%d'%pairs['ih2'][i],fontsize=10)
        # ax3.text((pairs['ra1'][i]+pairs['ra2'][i])/2.,(pairs['dec1'][i]+pairs['dec2'][i])/2.,'%d'%pairs[i]['ipair'],fontsize=10)
    cax=ax3.scatter(pairs['ra1'],pairs['dec1'], halo_size , c=halo1['z'] , marker = 'o' ,vmin=minz, vmax=maxz,zorder=2) 
    ax3.scatter(pairs['ra2'],pairs['dec2'], halo_size , c=halo2['z'] , marker = 'o' ,vmin=minz, vmax=maxz,zorder=2) 
    for f in range(len(fieldscat)):
        x1=fieldscat[f]['ra_min']
        x2=fieldscat[f]['de_min']
        l1=fieldscat[f]['ra_max'] - fieldscat[f]['ra_min']
        l2=fieldscat[f]['de_max'] - fieldscat[f]['de_min']
        rect = matplotlib.patches.Rectangle((x1,x2), l1, l2, facecolor=None, edgecolor='Black', linewidth=1.0 , fill=None)
        ax3.add_patch(rect)

    ax3.set_xlim(box_w3[0],box_w3[1])
    ax3.set_ylim(box_w3[2],box_w3[3])
    ax3.invert_xaxis()
    ax3.set_xticks([331,333,335])
    ax3.set_yticks([-1,2,5])
    ax3.tick_params(axis='both', which='major', labelsize=22)
    
    # ax4.scatter(cluscat_durret['ra'],cluscat_durret['de'],s=50,marker='d')
    # cax=ax4.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s')
    # ax4.scatter(pairs['ra1'],pairs['dec1'], halo_size , c=halo1['z'] , marker = 'o' ,vmin=minz, vmax=maxz)
    # cax=ax4.scatter(pairs['ra2'],pairs['dec2'], 60 , c=halo2['z'] , marker = 'o' ,vmin=minz, vmax=maxz)
    # for i in range(len(pairs)): 
    #     ax4.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c='r')
    #     ax4.text(pairs['ra1'][i],pairs['dec1'][i],'%d'%pairs['ih1'][i],fontsize=10)
    #     ax4.text(pairs['ra2'][i],pairs['dec2'][i],'%d'%pairs['ih2'][i],fontsize=10)
    #     ax3.text((pairs['ra1'][i]+pairs['ra2'][i])/2.,(pairs['dec1'][i]+pairs['dec2'][i])/2.,'%d'%pairs[i]['ipair'],fontsize=10)
    # for f in range(len(fieldscat)):
    #     x1=fieldscat[f]['ra_min']
    #     x2=fieldscat[f]['de_min']
    #     l1=fieldscat[f]['ra_max'] - fieldscat[f]['ra_min']
    #     l2=fieldscat[f]['de_max'] - fieldscat[f]['de_min']
    #     rect = matplotlib.patches.Rectangle((x1,x2), l1, l2, facecolor=None, edgecolor='Black', linewidth=1.0 , fill=None)
    #     ax4.add_patch(rect)

    # ax4.set_xlim(box_w4[0],box_w4[1])
    # ax4.set_ylim(box_w4[2],box_w4[3])

    
    cbar_ax = fig.add_axes([0.9, 0.175, 0.015, 0.7])
    cbar=fig.colorbar(cax,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=20) 
    # cbar.set_ticks([0.3,0.35,0.4,0.45])
    # cbar.set_ticklabels([0.3,0.35,0.4])
    pl.figtext(0.93,0.52,'z',fontsize=22)
    # fig.suptitle('%d pairs - class %d' % (n_pairs_used, classif))
    pl.subplots_adjust(bottom=0.15)

    filename_fig = 'filament_map.png'
    # pl.savefig(filename_fig)
    # log.info('saved %s' , filename_fig)
    fig.show()
    # fig.close()
    import pdb; pdb.set_trace()

def figure_random():

    config['use_random_halos'] = True
    import filaments_analyse
    filaments_analyse.config=config
    filaments_analyse.args=args


    filename_shears = config['filename_shears']                                  # args.filename_shears 
    filename_pairs = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')
    # filename_pairs = config['filename_pairs'].replace('.fits','.addstats.fits')
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)
    print 'n_pairs' , n_pairs

    if 'cfhtlens' in filename_pairs:
        bins_snr_edges = [5,20]
        mass_param_name = 'snr'
    else:
        bins_snr_edges = [1e14,1e15]
        mass_param_name = 'm200'
    # bins_snr_centers = [ 3 , 6]
    bins_snr_centers = plotstools.get_bins_centers(bins_snr_edges)

    ids=np.nonzero(pairs['analysis']==1)[0]
    args.first=0
    args.num=-1

    n_set = 19
    n_reps = 32
    n_pairs_use = n_set*n_reps

    list_prod_2D = []

    use_pickle=args.use_pickle

    ids_used = []
    current_id = 0

    filename_pickle_nulltest = args.filename_config.replace('.yaml','.nulltest.pp2')
    if use_pickle:
        dic=tabletools.loadPickle(filename_pickle_nulltest)
        sum_pdf = dic['sum_pdf']
        grid_dict = dic['grid_dict']
        list_prod_2D = dic['list_prod_2D']
        prod_pdf_all = dic['prod_pdf_all']

    list_nsig1 = []    
    list_nsig2 = []    

    for ir in range(n_reps):

        if use_pickle: break

        print 'boot % 3d  ------------- ' % (ir)

        ids = []

        while len(ids) < n_set:

            # id_try = np.random.choice(1280)
            id_try = current_id
            if id_try not in ids_used:

                filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir, id_try, id_try+1 , name_data)
                if os.path.isfile(filename_pickle):
                    ids.append(id_try)
                    ids_used.append(id_try)
                else:
                    print 'file not found: %s' % filename_pickle
                current_id+=1

        print ids
        prod_pdf, grid_dict, list_ids_used , n_pairs_used = filaments_analyse.get_prob_prod_gridsearch_2D(ids , plots=False)

        max0, max1 = np.unravel_index(prod_pdf.argmax(), prod_pdf.shape)
        id_radius=max1
        at_radius=grid_dict['grid_radius'][0,id_radius]
        print '--- using radius=' , at_radius
        kappa_at_radius=prod_pdf[:,id_radius].copy()
        kappa_at_radius/=np.sum(kappa_at_radius)
        kappa_grid = grid_dict['grid_kappa0'][:,id_radius]
        max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(kappa_grid,kappa_at_radius)
        print '--- %2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)
        list_nsig1.append( max_par/err_lo )

        id_kappa0=max0
        at_kappa0=grid_dict['grid_kappa0'][id_kappa0,0]
        print '--- using kappa0=' , at_kappa0
        radius_at_kappa=prod_pdf[id_kappa0,:].copy()
        radius_at_kappa/=np.sum(radius_at_kappa)
        radius_grid = grid_dict['grid_radius'][id_kappa0,:]
        max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(radius_grid,radius_at_kappa)
        print '--- %2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)
        list_nsig2.append( max_par/err_lo )

        list_prod_2D.append(prod_pdf)


        pl.figure()
        pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf)
        contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod_pdf,list_sigmas=[1,2])
        pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='r',lw=3)
        pl.axis('tight')
        filename_fig = 'figs/%s.%02d.png' % (args.filename_config.replace('.yaml',''),ir)
        pl.savefig(filename_fig)
        # pl.show()
        print 'saved %s' % filename_fig


    current_id = 0
    ids_used=[]
    for ir in range(1):

        if use_pickle: break

        print 'boot % 3d  ------------- ' % (ir)

        ids = []

        while len(ids) < n_pairs_use:
            print current_id

            # id_try = np.random.choice(1280)
            id_try = current_id
            if id_try not in ids_used:

                filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir, id_try, id_try+1 , name_data)
                if os.path.isfile(filename_pickle):
                    ids.append(id_try)
                    ids_used.append(id_try)
                else:
                    print 'file not found: %s' % filename_pickle
                current_id+=1

        print ids
        prod_pdf_all, grid_dict, list_ids_used , n_pairs_used = filaments_analyse.get_prob_prod_gridsearch_2D(ids , plots=False)



    if not use_pickle:
        sum_pdf = np.zeros_like(prod_pdf)
        for lp2D in list_prod_2D: sum_pdf += lp2D
        sum_pdf = sum_pdf / np.sum(sum_pdf.flatten())

    ent=[]
    for lp2D in list_prod_2D:
        ent_this= -(np.sum(lp2D.flatten() * np.log(lp2D.flatten())))/len(lp2D.flatten())
        ent.append(ent_this)
        print ent_this
    print 'mean ent', np.mean(np.array(ent))

    contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(sum_pdf,list_sigmas=[1,2])


    xlabel=r'$\Delta\Sigma$  $10^{14} \mathrm{M}_{\odot} \mathrm{Mpc^{-2} h}$'
    ylabel=r'R_{\mathrm{scale}} \ \ Mpc/h$'
    if config['kappa_is_K']:
            # xlabel=r' $\Delta\Sigma^{face-on}$ /   $ \mathrm{mean}(\Delta\Sigma_{200})}$ '
            # xlabel=r'$\frac{ \Delta\Sigma_{\mathrm{face-on}}^{\mathrm{fil}} }{ 0.5 (\Delta\Sigma_{200}^{\mathrm{halo1}}+\Delta\Sigma_{200}^{\mathrm{halo2}} )/2}$'
            # ylabel=r'$\frac{ R_{c}^{\mathrm{fil}} }{ (R_{200}^{\mathrm{halo1}}+R_{200}^{\mathrm{halo2}} )/2}$'
            xlabel=r'$ D_{f} = \Delta\Sigma_{\mathrm{peak}}^{\mathrm{filament}} /  \Delta\Sigma_{R200}^{\mathrm{halos}}  $'
            ylabel=r'$R_{\mathrm{scale}} \ \ \mathrm{Mpc/h}$'


    # normal plot
    pl.figure()
    pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf)
    pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,levels=contour_levels,colors='y')
     
    # # paper plots in ugly colormap
    pl.figure(figsize=(8,6))
    cp1 = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,levels=contour_levels,colors='b',linewidths=3)
    cp2 = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf_all,levels=contour_levels,colors='b',linewidths=3,linestyles='dashed')
    fmt = {}; strs = [ r'$68\%$', r'$95\%$'] ; 
    # fmt = {}; strs = [ '', '', r'$99\%$'] ; 
    for l,s in zip( cp1.levels, strs ): fmt[l] = s
    manual_locations = [(0.0,1),(0.34,1.26)]
    pl.clabel(cp1, cp1.levels, fmt=fmt , fontsize=20, manual=manual_locations)
    for l,s in zip( cp1.levels, strs ): fmt[l] = s
    manual_locations = [(0.0,1),(0.1,0.26)]
    pl.clabel(cp2, cp2.levels, fmt=fmt , fontsize=20, manual=manual_locations)
    # pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,cmap=pl.cm.YlOrRd)
    cp1 = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,levels=[contour_levels[0],1], alpha=0.2 ,  colors=['b'])
    cp1 = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,levels=[contour_levels[1],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,levels=[contour_levels[2],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,levels=[contour_levels[3],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,levels=[contour_levels[4],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,levels=[contour_levels[0],contour_levels[2]], alpha=0.25 ,  cmap=pl.cm.Blues)
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],sum_pdf,levels=contour_levels, alpha=0.25 ,  cmap=pl.cm.bone)
    pl.xlabel(xlabel,fontsize=28,labelpad=5)
    pl.ylabel(ylabel,fontsize=28,labelpad=1)
    # pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    # pl.plot(max_kappa0,max_radius,'b+',markersize=20,lw=50)
    pl.axis('tight')
    pl.yticks([0.5,1,1.5,2])
    # pl.ylim([0,3])
    pl.ylim([0,2])
    pl.xticks([0.0,0.125,0.25,0.375,0.5])
    pl.tick_params(axis='both', which='major', labelsize=20)
    # pl.xlim([0,0.8])
    pl.xlim([0,0.5])
    pl.subplots_adjust(bottom=0.18,left=0.15,top=0.95)



    # pl.figure()
    # for ir in range(0,200,25): pl.plot(grid_dict['grid_kappa0'][:,ir],lp2D[:,ir],label='%2.2f'%grid_dict['grid_radius'][0,ir]); pl.title(grid_dict['grid_radius'][0,ir]); pl.legend()
    # for ir in range(0,200,25): max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_dict['grid_kappa0'][:,ir],lp2D[:,ir]); print grid_dict['grid_radius'][0,ir] , max_par , err_hi , err_lo
    # for ir in range(len(list_prod_2D)): 
    #     list_prod_2D[ir].argmax(axis=)
    pl.show()

    dic={'sum_pdf':sum_pdf,'grid_dict':grid_dict,'list_prod_2D':list_prod_2D,'prod_pdf_all':prod_pdf_all}
    tabletools.savePickle(filename_pickle_nulltest,dic)

    import pdb; pdb.set_trace()

def figure_contours():

    import filaments_analyse
    filaments_analyse.config=config
    filaments_analyse.args=args


    filename_shears = config['filename_shears']                                  # args.filename_shears 
    filename_pairs = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')
    # filename_pairs = config['filename_pairs'].replace('.fits','.addstats.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)
    print 'n_pairs' , n_pairs

    if 'cfhtlens' in filename_pairs:
        bins_snr_edges = [5,20]
        mass_param_name = 'snr'
    else:
        bins_snr_edges = [1e14,1e15]
        mass_param_name = 'm200'
    # bins_snr_centers = [ 3 , 6]
    bins_snr_centers = plotstools.get_bins_centers(bins_snr_edges)

    ids=np.nonzero(pairs['analysis']==1)[0]
    args.first=0
    args.num=-1
    prod_pdf, grid_dict, list_ids_used , n_pairs_used = filaments_analyse.get_prob_prod_gridsearch_2D(ids)

    print 'used %d pairs' % n_pairs_used

    for ic in list_ids_used:
        print 'ipair=% 5d ih1=% 5d ih2=% 5d m200_h1=%5.2e m200_h2=%5.2e' % (ic, pairs[ic]['ih1'] , pairs[ic]['ih2'], pairs[ic]['m200_h1_fit'], pairs[ic]['m200_h2_fit'])

    res_dict = { 'prob' : prod_pdf , 'params' : grid_dict, 'n_obj' : n_pairs_used }

    # single 2d plot

    max0, max1 = np.unravel_index(prod_pdf.argmax(), prod_pdf.shape)
    print grid_dict['grid_kappa0'][max0,max1], grid_dict['grid_radius'][max0,max1]
    max_radius = grid_dict['grid_radius'][0,max1]
    max_kappa0 = grid_dict['grid_kappa0'][max0,0]
    contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod_pdf,list_sigmas=[1,2])
    xlabel=r'$\Delta\Sigma$  $10^{14} \mathrm{M}_{\odot} \mathrm{Mpc}^{-2} h$'
    ylabel=r'radius $\mathrm{Mpc}/h$'
    if config['kappa_is_K']:
            # xlabel=r' $\Delta\Sigma^{face-on}$ /   $ \mathrm{mean}(\Delta\Sigma_{200})}$ '
            # xlabel=r'$\frac{ \Delta\Sigma_{\mathrm{face-on}}^{\mathrm{fil}} }{ 0.5 (\Delta\Sigma_{200}^{\mathrm{halo1}}+\Delta\Sigma_{200}^{\mathrm{halo2}} )/2}$'
            # ylabel=r'$\frac{ R_{c}^{\mathrm{fil}} }{ (R_{200}^{\mathrm{halo1}}+R_{200}^{\mathrm{halo2}} )/2}$'
            xlabel=r'$ D_{f} =  \Delta\Sigma_{\mathrm{peak}}^{\mathrm{filament}} /  \Delta\Sigma_{R200}^{\mathrm{halos}}  $'
            # ylabel=r'$ R^{f} =  R_{\mathrm{scale}}^{\mathrm{filament}} /  R_{200}^{\mathrm{halos}} $'
            ylabel=r'$ R_{\mathrm{scale}} \ \ \ \mathrm{Mpc/h}$'

    # ==============================================================================
    # 2D plot for thesis
    # ==============================================================================

    
    pl.figure()
    cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='y')
    pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf)
    pl.axhline(max_radius,color='r')
    pl.axvline(max_kappa0,color='r')
    pl.xlabel(xlabel,fontsize=24)
    pl.ylabel(ylabel,fontsize=24)
    # pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    pl.axis('tight')

    # paper plots in ugly colormap
    pl.figure(figsize=(16,12))
    cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='b',linewidths=3)
    fmt = {}; strs = [ r'$68\%$', r'$95\%$'] ; 
    # fmt = {}; strs = [ '', '', r'$99\%$'] ; 
    for l,s in zip( cp.levels, strs ): fmt[l] = s
    manual_locations = [(0.5,2.2),(0.75,2.0)]
    pl.clabel(cp, cp.levels, fmt=fmt , fontsize=24, manual=manual_locations)
    # pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,cmap=pl.cm.YlOrRd)
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],1], alpha=0.2 ,  colors=['b'])
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[1],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[2],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[3],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[4],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],contour_levels[2]], alpha=0.25 ,  cmap=pl.cm.Blues)
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels, alpha=0.25 ,  cmap=pl.cm.bone)



    pl.xlabel(xlabel,fontsize=38)
    pl.ylabel(ylabel,fontsize=38)
    # pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    # pl.plot(max_kappa0,max_radius,'b+',markersize=20,lw=50)
    pl.scatter(max_kappa0,max_radius,200,c='b',marker='+')
    pl.axis('tight')
    pl.yticks([1,2,3,4])
    # pl.ylim([0,3])
    pl.xticks([0.0,0.5,1.,1.5,2.0])
    # pl.xlim([0,0.8])
    xmin,xmax = 0,1.2
    ymin,ymax = 0,4.5
    pl.xlim([xmin,xmax])
    pl.ylim([ymin,ymax])
    pl.tick_params(axis='both', which='major', labelsize=22)

    # m1 = 0.0878
    # m2 = 0.0878
    # ax2 = pl.gca().twiny()
    # ax2.plot([xmin*m1,xmax*m1],[-1,-1] ,'r')
    # ax2.set_xticks([0.05,0.1,0.15])
    # # ax2.xscale('log')
    # ax2.tick_params(axis='both', which='major', labelsize=22)
    # ax2.set_xlabel(r'$\Delta\Sigma_{\mathrm{peak}}^{\mathrm{filament}} \ \  \mathrm{M_{\odot} \ Mpc^{-2} h} \ \  \mathrm{for} \ \ \mathrm{M_{200}^{halos}}=10^{14} \ \ \mathrm{M_{\odot}/h}$',fontsize=38)

    pl.subplots_adjust(bottom=0.12,left=0.1,top=0.95)

    pl.figure()
    cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='y')
    pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf)
    pl.axhline(max_radius,color='r')
    pl.axvline(max_kappa0,color='r')
    pl.xlabel(xlabel,fontsize=24)
    pl.ylabel(ylabel,fontsize=24)
    # pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    pl.axis('tight')


    # ==============================================================================
    # 2D plot for paper
    # ==============================================================================

    # paper plots in ugly colormap
    pl.figure(figsize=(8,6))
    cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='b',linewidths=3)
    fmt = {}; strs = [ r'$68\%$', r'$95\%$'] ; 
    # fmt = {}; strs = [ '', '', r'$99\%$'] ; 
    for l,s in zip( cp.levels, strs ): fmt[l] = s
    manual_locations = [(0.5,2.2),(0.75,2.0)]
    pl.clabel(cp, cp.levels, fmt=fmt , fontsize=24, manual=manual_locations)
    # pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,cmap=pl.cm.YlOrRd)
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],1], alpha=0.2 ,  colors=['b'])
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[1],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[2],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[3],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[4],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],contour_levels[2]], alpha=0.25 ,  cmap=pl.cm.Blues)
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels, alpha=0.25 ,  cmap=pl.cm.bone)



    pl.xlabel(xlabel,fontsize=38)
    pl.ylabel(ylabel,fontsize=38)
    # pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    # pl.plot(max_kappa0,max_radius,'b+',markersize=20,lw=50)
    pl.scatter(max_kappa0,max_radius,200,c='b',marker='+')
    pl.axis('tight')
    pl.yticks([1,2,3,4])
    # pl.ylim([0,3])
    pl.xticks([0.0,0.5,1.,1.5,2.0])
    # pl.xlim([0,0.8])
    xmin,xmax = 0,1.2
    ymin,ymax = 0,4.5
    pl.xlim([xmin,xmax])
    pl.ylim([ymin,ymax])
    pl.tick_params(axis='both', which='major', labelsize=22)

    # m1 = 0.0878
    # m2 = 0.0878
    # ax2 = pl.gca().twiny()
    # ax2.plot([xmin*m1,xmax*m1],[-1,-1] ,'r')
    # ax2.set_xticks([0.05,0.1,0.15])
    # # ax2.xscale('log')
    # ax2.tick_params(axis='both', which='major', labelsize=22)
    # ax2.set_xlabel(r'$\Delta\Sigma_{\mathrm{peak}}^{\mathrm{filament}} \ \  \mathrm{M_{\odot} \ Mpc^{-2} h} \ \  \mathrm{for} \ \ \mathrm{M_{200}^{halos}}=10^{14} \ \ \mathrm{M_{\odot}/h}$',fontsize=38)

    pl.subplots_adjust(bottom=0.2,left=0.15,top=0.95)

    
    # ===================================================================================
    # plot 1d - just kappa0
    # ===================================================================================

    prob_kappa0 = np.sum(prod_pdf,axis=1)
    prob_radius = np.sum(prod_pdf,axis=0)
    grid_kappa0 = grid_dict['grid_kappa0'][:,0] 
    grid_radius = grid_dict['grid_radius'][0,:] 
    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_kappa0,prob_kappa0)
    print 'kappa0 %2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_radius,prob_radius)
    print 'radius %2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

    id_radius=max1
    at_radius=grid_dict['grid_radius'][0,id_radius]
    print 'using radius=' , at_radius
    kappa_at_radius=prod_pdf[:,id_radius].copy()
    kappa_at_radius/=np.sum(kappa_at_radius)
    kappa_grid = grid_dict['grid_kappa0'][:,id_radius]
    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(kappa_grid,kappa_at_radius)
    print '%2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)


    pl.figure(figsize=(8,6))
    pl.plot(kappa_grid,kappa_at_radius,'b-',label=r'$R_{s}$=%2.2f'%at_radius,lw=4)
    pl.plot(kappa_grid,prob_kappa0,'b--',label=r'$R_{s}$ marginalised',lw=4)
    # pl.title('CFHTLens + BOSS-DR10, radius=%2.2f, using %d pairs' % (at_radius,n_pairs_used))
    pl.xlabel(xlabel,fontsize=38)
    pl.yticks([])
    pl.xticks([0,0.5,1,1.5,2.0])
    pl.xlim([0,1.2])
    pl.ylim([0,0.008])
    pl.tick_params(axis='both', which='major', labelsize=22)
    pl.legend(ncol=2,prop={'size':21},mode='expand')
    pl.subplots_adjust(bottom=0.2)


    id_kappa0=max0
    at_kappa0=grid_dict['grid_kappa0'][id_kappa0,0]
    print 'using kappa0=' , at_kappa0
    radius_at_kappa=prod_pdf[id_kappa0,:].copy()
    radius_at_kappa/=np.sum(radius_at_kappa)
    radius_grid = grid_dict['grid_radius'][id_kappa0,:]
    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(radius_grid,radius_at_kappa)
    print '%2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

    log_max_model = np.log(prod_pdf[max0,max1])
    log_max_null = np.log(prod_pdf[0,0])
    D = 2*(log_max_model - log_max_null)
    LRT_pval = 1. - scipy.stats.chi2.cdf(D, 2)
    print 'LRT_pval' , LRT_pval
    print 'ppf', scipy.stats.norm.ppf(LRT_pval, loc=0, scale=1)


    pl.figure(figsize=(8,6))
    pl.plot(radius_grid,radius_at_kappa,'b-',label=r'$D_{f}$=%2.2f'%at_kappa0,lw=4)
    pl.plot(radius_grid,prob_radius,'b--',label=r'$D_{f}$ marginalised',lw=4)
    pl.xlabel(ylabel,fontsize=38)
    pl.yticks([])
    pl.xticks([0,1,2,3,4])
    pl.xlim([0,4.5])
    pl.ylim([0,0.006])
    pl.legend(ncol=2)
    pl.tick_params(axis='both', which='major', labelsize=22)
    pl.legend(ncol=2,prop={'size':21},mode='expand')
    pl.subplots_adjust(bottom=0.2)
    # pl.title('CFHTLens + BOSS-DR10, radius=%2.2f, using %d pairs' % (at_kappa0,n_pairs_used))
    pl.show()

    total_normalisation = grid_dict['total_normalisation']
    print 'enthropy' , -(np.sum(prod_pdf.flatten() * np.log(prod_pdf.flatten())))/len(prod_pdf.flatten())

    import pdb; pdb.set_trace()

def table_individual():

    global halos ,filename_halos_cfhtlens ,filename_cluscat ,filename_fields ,bossdr10 ,pairs ,halo1 ,halo2 ,cluscat ,fieldscat ,filename_cluscat_durret ,cluscat_durret

    # candidates
    select= pairs['analysis']==1
    print 'candidates' , len(pairs[select])

    for ic,vc in enumerate(pairs[select]):

        h1= halos[vc['ih1']]
        h2= halos[vc['ih2']]

        print 'ic=%03d ra_1=% 10.5f dec_1=% 10.5f ra_2=% 10.5f dec_2=% 10.5f z_1=%2.4f z_2=%2.4f R_los=%2.2f R_pair=%10.f M1_fit=%2.2f M1_sig=%2.2f M2_fit=%2.2f M2_sig=%2.2f' % (
                    vc['ipair'],
                    vc['ra1'],
                    vc['dec1'],
                    vc['ra2'],
                    vc['dec2'],
                    h1['z'],
                    h2['z'],
                    vc['drloss'],
                    vc['R_pair'],
                    h1['m200_fit']/1e14,
                    h1['m200_sig'],
                    h2['m200_fit']/1e14,
                    h2['m200_sig'],
                    )

    for ic,vc in enumerate(pairs[select]):

        h1= halos[vc['ih1']]
        h2= halos[vc['ih2']]

        print '$% 4d$ & $% 12.3f$ & $% 12.3f$ & $% 12.3f$ & $% 12.3f$ & $%2.4f$ & $%2.4f$ & $%12.3f $ & $%12.3f$ & $%6.1f$ & $%6.1f$ & $%6.1f$ & $%6.1f$  \\\\' % (
                    ic+1,
                    vc['ra1'],
                    vc['dec1'],
                    vc['ra2'],
                    vc['dec2'],
                    halo1['z'][vc['ipair']],
                    halo2['z'][vc['ipair']],
                    vc['Dxy'],
                    vc['Dlos'],
                    h1['m200_fit']/1e14,
                    h1['m200_sig'],
                    h2['m200_fit']/1e14,
                    h2['m200_sig'],
                    )

def figure_prior():

    filename_prior = config['filename_halos'].replace('.fits','.prior.pp2')  
    prior_dict = tabletools.loadPickle(filename_prior,log=0)
    pairs = tabletools.loadTable(config['filename_pairs'],log=0)
    ids = pairs['analysis'] == 1
    halos_use = np.concatenate([pairs[ids]['ih1'],pairs[ids]['ih2']])
    prior_ids = prior_dict['halos_like'][halos_use]
    norm1 = prior_ids-np.kron( np.ones([1,prior_ids.shape[1]]) , prior_ids.max(axis=1)[:,None] )
    norm2 = np.exp(norm1)/np.kron( np.ones([1,prior_ids.shape[1]]) , np.sum(np.exp(norm1),axis=1)[:,None] )
    norm3 = np.sum(norm2,axis=0)
    import scipy.interpolate
    x=prior_dict['grid_M200']


    pl.figure(figsize=(8,6))
    pl.plot(x,norm3,lw=4)
    pl.xlabel(r'$M_{200} \ \ \mathrm{M_{\odot} / h}$',fontsize=30)
    # pl.xticks([1e14,2e14,3e14,4e14,5e14,6e14,7e14,8e14],[r'1^{'])
    # pl.xticks([1,2,3,4,5,6,7,8])
    pl.yticks([])
    pl.xlim([1e14,1e15])
    pl.ylim([0,0.065])
    pl.xscale('log')
    pl.fill_between(x, 0, norm3,alpha=0.2)
    pl.tick_params(axis='both', which='major', labelsize=22)
    pl.subplots_adjust(bottom=0.2)

    # pl.xscale('log')
    pl.show()

    import pdb; pdb.set_trace()



def main():


    valid_actions = ['figure_fields','figure_model','figure_contours','table_individual','figures_individual'  , 'figure_density' , 'figure_random' , 'figure_prior']

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )
    parser.add_argument('-rd','--results_dir', action='store', help='where results files are' , default='results/' )
    parser.add_argument('-hr','--halo_removal', action='store', default='prior', choices=('flat','prior','ml','exp' , 'default'), help='which halo removal method to use' )
    parser.add_argument('--use_pickle', action='store_true', default=False,  help='if to use existing pickle to make plots' )

    global args

    args = parser.parse_args()
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    log.setLevel(logging_level)

    global config 
    config = yaml.load(open(args.filename_config))

    global halos ,filename_halos_cfhtlens ,filename_cluscat ,filename_fields ,bossdr10 ,pairs ,halo1 ,halo2 ,cluscat ,fieldscat ,filename_cluscat_durret ,cluscat_durret
    halos = tt.load(config['filename_halos'])
    filename_halos_cfhtlens = os.environ['HOME'] + '/data/CFHTLens/CFHTLens_DR10_LRG/BOSSDR10LRG.fits'
    filename_cluscat = os.environ['HOME'] + '/data/CFHTLens/ClusterZ/clustersz.fits'
    filename_fields =  os.environ['HOME'] + '/data/CFHTLens/field_catalog.fits'
    bossdr10 = pyfits.getdata(filename_halos_cfhtlens)
    pairs = tabletools.loadTable(config['filename_pairs'])
    halo1 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos1.fits'))
    halo2 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos2.fits'))
    cluscat = tabletools.loadTable(filename_cluscat)
    fieldscat = tabletools.loadTable(filename_fields)
    filename_cluscat_durret = os.environ['HOME'] + '/data/CFHTLens/CFHTLS_wide_clusters_Durret2011/wide.fits'
    cluscat_durret = tabletools.loadTable(filename_cluscat_durret)

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



main()