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
import scipy.interpolate as interp
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
    param1 = 0.75
    param0 = 0.9
    param0_prop = 1.5

    list_new = []
    list_old = []
    list_pro = []
    list_m200 = []

    for ip,vp in enumerate(pairs):

        if vp['analysis']!=1:
            continue
        
        nh1 = nfw.NfwHalo()
        nh1.z_cluster= halo1['z'][ip]
        nh1.M_200    = halo1['m200_fit'][ip]
        nh1.concentr = nh1.get_concentr()
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
        filament_radius = param1 * (nh1.R_200+nh2.R_200)/2.

        DeltaSigma_at_R200_bug = (np.abs(h2g1[0]*h2_Sigma_crit)+np.abs(h1g1[0]*h1_Sigma_crit))/2.
        filament_kappa0_old = param0_old * DeltaSigma_at_R200_bug / 1e14

        filament_kappa0_prop = param0_prop * (-1)*(h1g1m[0]+h2g1m[0]) * h2_Sigma_critm / 1e14

        print '% 4d m1=%2.2e m2=%2.2e DS200=%2.2e kappa0=%2.3f kappa0_bug=%2.3f radius=%2.2f ' % (ip,nh1.M_200,nh2.M_200,DeltaSigma_at_R200,filament_kappa0,filament_kappa0_old,filament_radius)
        print '---- DSmid=%2.2e kappa0=%2.2f' % ((h1_DeltaSigmam[0]+h2_DeltaSigmam[0]),filament_kappa0)
        print '---- %5.3f %5.3e %5.3e %5.3f' % (h1g1[0], h1_DeltaSigma[0], h1_Sigma_crit, h1_kappa[0])
        print '---- %5.3f %5.3e %5.3e %5.3f' % (h2g1[0], h2_DeltaSigma[0], h2_Sigma_crit, h2_kappa[0])
        print '---- change: %2.2f' % (filament_kappa0_old/filament_kappa0)

        list_m200.append((nh2.M_200+nh1.M_200)/2.)
        list_old.append(filament_kappa0_old)        
        list_new.append(filament_kappa0)
        list_pro.append(filament_kappa0_prop)        

    list_m200 = np.array(list_m200)
    list_new = np.array(list_new)
    import fitting
    a,b,Cab = fitting.get_line_fit(list_m200/1e14,list_new,np.ones_like(list_new))
    print 'line fit a' ,  a
    print 'line fit b' ,  b
    print 'line fit C' ,  Cab

    # pl.plot(list_m200,list_old,'r.')
    pl.plot(list_m200/1e14,list_new,'g.')
    pl.plot(list_m200/1e14,b*list_m200/1e14+a)
    pl.xlabel('mean m200')
    pl.ylabel(r'mean \Delta\Sigma')
    # pl.plot(list_m200,list_pro,'b.')
    pl.show()



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

    param_radius = 0.75
    param_kappa0 = 1.2
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

    cmap = pl.get_cmap('Blues')
    pcm = pl.pcolormesh(shear_u_mpc,shear_v_mpc,model_DeltaSigma,cmap=cmap,norm=pl.matplotlib.colors.LogNorm())
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
    quiver_scale = 1.2
    nuse = 4
   
    shear_u_mpc = shear_u_mpc[::nuse,::nuse]
    shear_v_mpc = shear_v_mpc[::nuse,::nuse]
    shear_model_g1 = shear_model_g1[::nuse,::nuse]
    shear_model_g2 = shear_model_g2[::nuse,::nuse]
    emag = emag[::nuse,::nuse]
    ephi = ephi[::nuse,::nuse]


    hc = 8.42
    rad = 1.1 # mpc
    mask = (np.sqrt((shear_u_mpc-hc)**2 + shear_v_mpc**2) > rad) * (np.sqrt((shear_u_mpc+hc)**2 + shear_v_mpc**2) > rad) 
    shear_u_mpc = shear_u_mpc[mask]
    shear_v_mpc = shear_v_mpc[mask]
    shear_model_g1 = shear_model_g1[mask]
    shear_model_g2 = shear_model_g2[mask]
    emag = emag[mask]
    ephi = ephi[mask]
    

    pos=hc*1.015-np.pi/4.
    pl.plot([-pos,pos],[ param_radius, param_radius],c='k')
    pl.plot([-pos,pos],[-param_radius,-param_radius],c='k')

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
    pl.gca().add_patch(matplotlib.patches.Rectangle((-2.5,-3.7),5,1.1,facecolor='white', edgecolor='black'))
    pl.gca().add_patch(matplotlib.patches.Circle((-hc,0),1,edgecolor='k',facecolor='none',lw=1))
    pl.gca().add_patch(matplotlib.patches.Circle(( hc,0),1,edgecolor='k',facecolor='none',lw=1))
    qk = pl.quiverkey(quiv, 0.475, 0.22, 0.01, r'$g=0.01$', labelpos='E', coordinates='figure', fontproperties={'weight': 'bold' , 'size':14})

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
    cbar = pl.colorbar(pcm,cax=cbaxes, ticks=[1e13,1e14,1e15])
    # cbar = pl.colorbar(pcm,cax=cbaxes)
    pl.figtext(0.935,0.2,r'$\Delta\Sigma$')

    print 'fitobj.nh2.R_200' , fitobj.nh2.R_200

    # import pdb; pdb.set_trace()
    # print fitobj.nh2.r_200
    pl.show()


def figure_fields():

    global halos ,filename_halos_cfhtlens ,filename_cluscat ,filename_fields ,bossdr10 ,pairs ,halo1 ,halo2 ,cluscat ,fieldscat ,filename_cluscat_durret ,cluscat_durret

    box_w1 = [29.5,39.5,-12,-3]
    box_w2 = [208,221,50.5,58.5]
    box_w3 = [329.5,336,-2,5.5]
    box_w4 = [131.5,137.5,-6.5,-0.5]

    select = pairs['analysis']==1
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
    fig = pl.figure(figsize=(15,5))
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

    fig.text(0.5, 0.04, 'RA', ha='center', va='center')
    fig.text(0.06, 0.5, 'Dec', ha='center', va='center', rotation='vertical')

    minz=0.3
    maxz=0.45
    halo_size = 100

    # ax1.scatter(cluscat_durret['ra'],cluscat_durret['de'],s=50,marker='d', vmin=minz, vmax=maxz)
    # ax1.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s', vmin=minz, vmax=maxz)
    ax1.text(box_w1[1]-0.2,box_w1[2]+0.2,'W1')
    ax1.scatter(pairs['ra1'],pairs['dec1'], halo_size , c=halo1['z'] , marker = 'o' , vmin=minz, vmax=maxz) #
    ax1.scatter(pairs['ra2'],pairs['dec2'], halo_size , c=halo2['z'] , marker = 'o' , vmin=minz, vmax=maxz) #
    for i in range(len(pairs)): 
        ax1.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c='r')
        # ax1.text(pairs['ra1'][i],pairs['dec1'][i],'%d'%pairs['ih1'][i],fontsize=10)
        # ax1.text(pairs['ra2'][i],pairs['dec2'][i],'%d'%pairs['ih2'][i],fontsize=10)
        # ax1.text((pairs['ra1'][i]+pairs['ra2'][i])/2.,(pairs['dec1'][i]+pairs['dec2'][i])/2.,'%d'%pairs[i]['ipair'],fontsize=10)
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

    # ax2.scatter(cluscat_durret['ra'],cluscat_durret['de'],s=50,marker='d',vmin=minz, vmax=maxz)
    # ax2.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s',vmin=minz, vmax=maxz)
    ax2.text(box_w2[1]-0.2,box_w2[2]+0.2,'W3')
    ax2.scatter(pairs['ra1'],pairs['dec1'], halo_size , c=halo1['z'] , marker = 'o' ,vmin=minz, vmax=maxz) #
    ax2.scatter(pairs['ra2'],pairs['dec2'], halo_size , c=halo2['z'] , marker = 'o' ,vmin=minz, vmax=maxz) #      
    for i in range(len(pairs)): 
        ax2.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c='r')
        # ax2.text(pairs['ra1'][i],pairs['dec1'][i],'%d'%pairs['ih1'][i],fontsize=10)
        # ax2.text(pairs['ra2'][i],pairs['dec2'][i],'%d'%pairs['ih2'][i],fontsize=10)
        # ax2.text((pairs['ra1'][i]+pairs['ra2'][i])/2.,(pairs['dec1'][i]+pairs['dec2'][i])/2.,'%d'%pairs[i]['ipair'],fontsize=10)
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

    ax3.text(box_w3[1]-0.2,box_w3[2]+0.2,'W4')
    # ax3.scatter(cluscat_durret['ra'],cluscat_durret['de'],s=50,marker='d',vmin=minz, vmax=maxz)
    # ax3.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s',vmin=minz, vmax=maxz)
    cax=ax3.scatter(pairs['ra1'],pairs['dec1'], halo_size , c=halo1['z'] , marker = 'o' ,vmin=minz, vmax=maxz) 
    ax3.scatter(pairs['ra2'],pairs['dec2'], halo_size , c=halo2['z'] , marker = 'o' ,vmin=minz, vmax=maxz) 
    for i in range(len(pairs)): 
        ax3.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c='r')
        # ax3.text(pairs['ra1'][i],pairs['dec1'][i],'%d'%pairs['ih1'][i],fontsize=10)
        # ax3.text(pairs['ra2'][i],pairs['dec2'][i],'%d'%pairs['ih2'][i],fontsize=10)
        # ax3.text((pairs['ra1'][i]+pairs['ra2'][i])/2.,(pairs['dec1'][i]+pairs['dec2'][i])/2.,'%d'%pairs[i]['ipair'],fontsize=10)
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

    
    cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
    fig.colorbar(cax,cax=cbar_ax)
    # fig.suptitle('%d pairs - class %d' % (n_pairs_used, classif))

    filename_fig = 'filament_map.png'
    # pl.savefig(filename_fig)
    # log.info('saved %s' , filename_fig)
    fig.show()
    # fig.close()
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
            xlabel=r'$ \Delta\Sigma_{\mathrm{face-on}}^{\mathrm{filament}} /  \Delta\Sigma_{200}^{\mathrm{halos}}  $'
            ylabel=r'$ R_{c}^{\mathrm{filament}} /  R_{200}^{\mathrm{halos}} $'

    
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
    pl.figure()
    cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='b')
    fmt = {}; strs = [ r'$68\%$', r'$95\%$'] ; 
    # fmt = {}; strs = [ '', '', r'$99\%$'] ; 
    for l,s in zip( cp.levels, strs ): fmt[l] = s
    manual_locations = [(1.1,0.5),(1.5,0.5)]
    pl.clabel(cp, cp.levels, fmt=fmt , fontsize=12, manual=manual_locations)
    # pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,cmap=pl.cm.YlOrRd)
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],1], alpha=0.2 ,  colors=['b'])
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[1],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[2],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[3],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[4],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],contour_levels[2]], alpha=0.25 ,  cmap=pl.cm.Blues)
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels, alpha=0.25 ,  cmap=pl.cm.bone)
    pl.xlabel(xlabel,fontsize=20)
    pl.ylabel(ylabel,fontsize=20)
    # pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    # pl.plot(max_kappa0,max_radius,'b+',markersize=20,lw=50)
    pl.scatter(max_kappa0,max_radius,200,c='b',marker='+')
    pl.axis('tight')
    pl.yticks([1,2,3,4])
    # pl.ylim([0,3])
    pl.ylim([0,4])
    pl.xticks([0.0,0.5,1.,1.5,2.0])
    # pl.xlim([0,0.8])
    pl.xlim([0,2])
    pl.subplots_adjust(bottom=0.12,left=0.1,top=0.95)

    # plot 1d - just kappa0
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

    pl.figure()
    pl.plot(kappa_grid,kappa_at_radius)
    # pl.title('CFHTLens + BOSS-DR10, radius=%2.2f, using %d pairs' % (at_radius,n_pairs_used))
    pl.xlabel(xlabel)


    id_kappa0=max0
    at_kappa0=grid_dict['grid_kappa0'][id_kappa0,0]
    print 'using kappa0=' , at_kappa0
    radius_at_kappa=prod_pdf[id_kappa0,:].copy()
    radius_at_kappa/=np.sum(radius_at_kappa)
    radius_grid = grid_dict['grid_radius'][id_kappa0,:]
    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(radius_grid,radius_at_kappa)
    print '%2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

    pl.figure()
    pl.plot(radius_grid,radius_at_kappa)
    # pl.title('CFHTLens + BOSS-DR10, radius=%2.2f, using %d pairs' % (at_kappa0,n_pairs_used))
    pl.xlabel(xlabel)

    pl.show()

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



def main():


    valid_actions = ['figure_fields','figure_model','figure_contours','table_individual','figures_individual'  , 'figure_density']

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )
    parser.add_argument('-rd','--results_dir', action='store', help='where results files are' , default='results/' )

    # parser.add_argument('-d', '--dry', default=False,  action='store_true', help='Dry run, dont generate data'), len(remove_list), len(set(remove_list)

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
    halos = tabletools.loadTable(config['filename_halos'])
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