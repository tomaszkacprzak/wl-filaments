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

    id_pair = 2
    shears_info = tabletools.loadPickle(filename_shears,id_pair)

    import filaments_model_2hf
    import filament
    import nfw
    fitobj = filaments_model_2hf.modelfit()
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

    param_radius = 0.5
    param_kappa0 = 0.3
    param_masses = 14
    shear_model_g1, shear_model_g2, limit_mask = fitobj.draw_model([param_kappa0, param_radius, param_masses, param_masses,])


    nx = sum(np.isclose(fitobj.shear_u_mpc,fitobj.shear_u_mpc[0]))
    ny = sum(np.isclose(fitobj.shear_v_mpc,fitobj.shear_v_mpc[0]))
    print nx,ny
    shear_u_mpc = np.reshape(fitobj.shear_u_mpc,[nx,ny])
    shear_v_mpc = np.reshape(fitobj.shear_v_mpc,[nx,ny])
    shear_model_g1= np.reshape(shear_model_g1,[nx,ny])
    shear_model_g2= np.reshape(shear_model_g2,[nx,ny])
    limit_mask=np.reshape(limit_mask,[nx,ny])

    # import pdb; pdb.set_trace()
    # pl.figure()
    # pl.pcolormesh( shear_u_mpc , shear_v_mpc , shear_model_g1)   
        
    # pl.figure()
    # pl.pcolormesh( shear_u_mpc , shear_v_mpc , shear_model_g2)
    
    # pl.figure(figsize=(27.3/3.,10/3.))
    pl.figure(figsize=(26/3.,10/3.))
    pl.subplots_adjust(bottom=0.15)

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
    quiver_scale = 0.55
    nuse = 4

    shear_u_mpc = shear_u_mpc[::nuse,::nuse]
    shear_v_mpc = shear_v_mpc[::nuse,::nuse]
    shear_model_g1 = shear_model_g1[::nuse,::nuse]
    shear_model_g2 = shear_model_g2[::nuse,::nuse]
    emag = emag[::nuse,::nuse]
    ephi = ephi[::nuse,::nuse]

    hc = 6.85
    rad = 1.1 # mpc
    mask = (np.sqrt((shear_u_mpc-hc)**2 + shear_v_mpc**2) > rad) * (np.sqrt((shear_u_mpc+hc)**2 + shear_v_mpc**2) > rad) 
    shear_u_mpc = shear_u_mpc[mask]
    shear_v_mpc = shear_v_mpc[mask]
    shear_model_g1 = shear_model_g1[mask]
    shear_model_g2 = shear_model_g2[mask]
    emag = emag[mask]
    ephi = ephi[mask]
    

    pos=6.75-np.pi/4.
    pl.plot([-pos,pos],[ param_radius, param_radius],c='k')
    pl.plot([-pos,pos],[-param_radius,-param_radius],c='k')

    import matplotlib
    quiv=pl.quiver(shear_u_mpc,shear_v_mpc,emag*np.cos(ephi),emag*np.sin(ephi),linewidths=0.001,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='k',label='original',scale=quiver_scale , width = line_width)  
    # pl.gca().add_patch(matplotlib.patches.Rectangle((5.5,2.3),5,10,color='w'))
    # qk = pl.quiverkey(quiv, 0.72, 0.8, 0.005, r'$g=0.005$', labelpos='E', coordinates='figure', fontproperties={'weight': 'bold' , 'size':20})
    pl.gca().add_patch(matplotlib.patches.Rectangle((-2.5,-4),5,1.1,color='w'))
    pl.gca().add_patch(matplotlib.patches.Circle((-6.85,0),1,edgecolor='k',facecolor='w',lw=1))
    pl.gca().add_patch(matplotlib.patches.Circle(( 6.85,0),1,edgecolor='k',facecolor='w',lw=1))
    qk = pl.quiverkey(quiv, 0.47, 0.225, 0.005, r'$g=0.005$', labelpos='E', coordinates='figure', fontproperties={'weight': 'bold' , 'size':14})


    pl.axis('equal')
    pl.xlim([min(shear_u_mpc.flatten())-0.5,max(shear_u_mpc.flatten())+0.5])
    pl.ylim([min(shear_v_mpc.flatten())-0.5,max(shear_v_mpc.flatten())+0.5])
    pl.xlabel(unit)
    pl.ylabel(unit)

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

           
    select = (pairs['BF1']>1) & (pairs['BF2']>1) & (pairs['manual_remove']==0)

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
    ax1.text(box_w1[0]+0.2,box_w1[2]+0.2,'W1')
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

    # ax2.scatter(cluscat_durret['ra'],cluscat_durret['de'],s=50,marker='d',vmin=minz, vmax=maxz)
    # ax2.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s',vmin=minz, vmax=maxz)
    ax2.text(box_w2[0]+0.2,box_w2[2]+0.2,'W3')
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

    ax3.text(box_w3[0]+0.2,box_w3[2]+0.2,'W4')
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
    args.results_dir='results'
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

    thres = 1.
    print 'BF n_clean '        ,   sum( (pairs['manual_remove']==0) & (pairs['BF1']>thres) & (pairs['BF2']>thres) & ( (pairs['eyeball_class']==1) | (pairs['eyeball_class']==3) )  )
    print 'BF n_contaminating ',   sum( (pairs['manual_remove']==0) & (pairs['BF1']>thres) & (pairs['BF2']>thres) & ( (pairs['eyeball_class']==2) | (pairs['eyeball_class']==0) )  )

    mass= (10**pairs['m200_h1_fit']+10**pairs['m200_h2_fit'])/2.
    select = (pairs['BF1']>thres) & (pairs['BF2']>thres) & (pairs['manual_remove']==0)
    # select = (pairs['BF1']>thres) & (pairs['BF2']>thres) & (pairs['eyeball_class']==1)
    # select = pairs['eyeball_class'] ==1
    ids=np.arange(n_pairs)[select]
    prod_pdf, grid_dict, list_ids_used , n_pairs_used = filaments_analyse.get_prob_prod_gridsearch_2D(ids)

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
    
    pl.figure()
    cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='y')
    pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf)
    pl.axhline(max_radius,color='r')
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    pl.axis('tight')

    # paper plots in ugly colormap
    pl.figure()
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
    # pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    pl.axis('tight')
    pl.ylim([0,4])

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


    pl.show()

def table_individual():

    global halos ,filename_halos_cfhtlens ,filename_cluscat ,filename_fields ,bossdr10 ,pairs ,halo1 ,halo2 ,cluscat ,fieldscat ,filename_cluscat_durret ,cluscat_durret

    # candidates
    select= (pairs['BF1']>1) & (pairs['BF2']>1) & (pairs['manual_remove']==0)
    print 'candidates' , len(pairs[select])

    import pdb; pdb.set_trace()
    for ic,vc in enumerate(pairs[select]):

        import scipy.special
        sigmas=scipy.special.erfinv(1.-vc['MLRT1'])*np.sqrt(2.)
        print 'ic=%03d ra_1=% 10.5f dec_1=% 10.5f ra_2=% 10.5f dec_2=% 10.5f z_1=%2.4f z_2=%2.4f R_los=%2.2f MLRT1=%10.4f sigmas=%2.2f BF1=%10.2f BF2=%10.2f R_pair=%10.f M_1=%2.2f M_2=%2.2f ML_radius=%2.2f ML_kappa0=%2.2f' % (
                    vc['ipair'],
                    vc['ra1'],
                    vc['dec1'],
                    vc['ra2'],
                    vc['dec2'],
                    halo1['z'][vc['ipair']],
                    halo2['z'][vc['ipair']],
                    vc['drloss'],
                    vc['MLRT1'],
                    sigmas,
                    vc['BF1'],
                    vc['BF2'],
                    vc['R_pair'],
                    vc['m200_h1_fit'],
                    vc['m200_h2_fit'],
                    vc['ML_radius'],
                    vc['ML_kappa0']
                    )

    for ic,vc in enumerate(pairs[select]):

        import scipy.special
        sigmas=scipy.special.erfinv(1.-vc['MLRT1'])*np.sqrt(2.)
        print '$% 4d$ & $% 12.3f$ & $% 12.3f$ & $% 12.3f$ & $% 12.3f$ & $%2.2f$ & $% 12.3f$ & $%12.3f $ & $%2.2f$ & $%12.4f$ & $%12.2f$ & $%12.2f$ & $%12.2f$ & $%2.2f$ & $%2.2f$ & $%2.2f$ & $%2.2f$ \\\\' % (
                    # vc['ipair'],
                    ic+1,
                    vc['ra1'],
                    vc['dec1'],
                    vc['ra2'],
                    vc['dec2'],
                    vc['R_pair'],
                    halo1['z'][vc['ipair']],
                    halo2['z'][vc['ipair']],
                    vc['drloss'],
                    vc['MLRT1'],
                    sigmas,
                    vc['BF1'],
                    vc['BF2'],
                    vc['m200_h1_fit'],
                    vc['m200_h2_fit'],
                    vc['ML_radius'],
                    vc['ML_kappa0']
                    )


    # significant
    cl = 1- 0.954499736
    select= (pairs['BF1']>1) & (pairs['BF2']>3) & (pairs['MLRT1']<cl) & (pairs['manual_remove']==0)
    print  'significant' , len(pairs[select])

    for ic,vc in enumerate(pairs[select]):

        import scipy.special
        sigmas=scipy.special.erfinv(1.-vc['MLRT1'])*np.sqrt(2.)
        print 'ic=%03d ra_1=% 10.5f dec_1=% 10.5f ra_2=% 10.5f dec_2=% 10.5f z_1=%2.4f z_2=%2.4f MLRT1=%10.4f sigmas=%2.2f BF1=%10.2f BF2=%10.2f R_pair=%10.f M_1=%2.2f M_2=%2.2f ML_radius=%2.2f ML_kappa0=%2.2f' % (
                    vc['ipair'],
                    vc['ra1'],
                    vc['dec1'],
                    vc['ra2'],
                    vc['dec2'],
                    halo1['z'][vc['ipair']],
                    halo2['z'][vc['ipair']],
                    vc['MLRT1'],
                    sigmas,
                    vc['BF1'],
                    vc['BF2'],
                    vc['R_pair'],
                    vc['m200_h1_fit'],
                    vc['m200_h2_fit'],
                    vc['ML_radius'],
                    vc['ML_kappa0']
                    )


def figure_vary_bf():


    import filaments_analyse
    filaments_analyse.config=config
    args.results_dir='results'
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

    thres = 1.
    print 'BF n_clean '        ,   sum( (pairs['manual_remove']==0) & (pairs['BF1']>thres) & (pairs['BF2']>thres) & ( (pairs['eyeball_class']==1) | (pairs['eyeball_class']==3) )  )
    print 'BF n_contaminating ',   sum( (pairs['manual_remove']==0) & (pairs['BF1']>thres) & (pairs['BF2']>thres) & ( (pairs['eyeball_class']==2) | (pairs['eyeball_class']==0) )  )

    # mass= (10**pairs['m200_h1_fit']+10**pairs['m200_h2_fit'])/2.
    # select = (pairs['BF1']>thres) & (pairs['BF2']>thres) & (pairs['manual_remove']==0)

    # levels = [1,1.1,1.2,1.3,1.4,1.5,2.0,3.0,4.0,]
    levels = [1.0,2.0,3.0,4.0,]
    list_ml_kappa0 = []
    list_ml_radius = []
    for ilvl,vlvl in enumerate(levels) :

        select = (pairs['BF1']>vlvl) & (pairs['BF2']>vlvl) & (pairs['manual_remove']==0)
        ids=np.arange(n_pairs)[select]
        prod_pdf, grid_dict, list_ids_used , n_pairs_used = filaments_analyse.get_prob_prod_gridsearch_2D(ids)

        print 'used %d pairs' % n_pairs_used     
        contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod_pdf,list_sigmas=[1,2,3,4,5])

        prob_kappa0 = np.sum(prod_pdf,axis=1)
        prob_radius = np.sum(prod_pdf,axis=0)
        grid_kappa0 = grid_dict['grid_kappa0'][:,0] 
        grid_radius = grid_dict['grid_radius'][0,:] 
        max_kappa0 , err_kappa0_hi , err_kappa0_lo = mathstools.estimate_confidence_interval(grid_kappa0,prob_kappa0)
        print 'kappa0 %2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_kappa0 , err_kappa0_hi , err_kappa0_lo, max_kappa0/err_kappa0_lo)

        max_radius , err_radius_hi , err_radius_lo = mathstools.estimate_confidence_interval(grid_radius,prob_radius)
        print 'radius %2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_radius , err_radius_hi , err_radius_lo, max_radius/err_radius_lo)

        list_ml_kappa0.append([max_kappa0 , err_kappa0_hi , err_kappa0_lo, n_pairs_used])
        list_ml_radius.append([max_radius , err_radius_hi , err_radius_lo , n_pairs_used])

        pl.figure(figsize=(10,10))
        cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='y')
        pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf)
        pl.title('n=%d kappa0=%2.4f +/- (%2.4f,%2.4f) radius=%2.2f +/- (%2.2f,%2.2f)' % (n_pairs_used,max_kappa0,err_kappa0_hi,err_kappa0_lo,max_radius,err_radius_hi,err_radius_lo))
        pl.axis('tight')
        # pl.show()

    arr_ml_kappa0=np.array(list_ml_kappa0)
    arr_ml_radius=np.array(list_ml_radius)

    xlim = [0.75,4.25]
    fig, ax1 = pl.subplots()

    ax1.errorbar(levels,arr_ml_kappa0[:,0]-0.05,yerr=arr_ml_kappa0[:,2],color='b',fmt='.')
    ax1.set_xlabel('Bayes Factor threshold')
    ax1.set_ylabel(r'$\Delta\Sigma$  $10^{14} \mathrm{M}_{\odot} \mathrm{Mpc}^{-2} h$',color='b')
    for tl in ax1.get_yticklabels():    tl.set_color('b')
    # ax1.axis["left"].label.set_color('blue')
    # ax1.axis["right"].label.set_color('red')

    ax2=ax1.twiny()
    ax2.plot(levels,arr_ml_kappa0[:,0]*0)
    ax2.set_xticks(levels)
    ax2.set_xticklabels(list(arr_ml_kappa0[:,3].astype('i4')))
    ax2.set_xlabel('number of included filaments')
    ax2.set_ylabel('',color='b')


    ax3=ax1.twinx()
    ax3.errorbar(np.array(levels)+0.05,arr_ml_radius[:,0],yerr=arr_ml_radius[:,2],color='r',fmt='.')
    # ax3.spines['left'].set_color('blue')
    # ax3.spines['right'].set_color('red')
    ax3.set_ylabel('radius Mpc/h',color='r')
    for tl in ax3.get_yticklabels():    tl.set_color('r')

    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax3.set_xlim(xlim)
    pl.show()
    
    pl.figure()
    pl.errorbar(levels,arr_ml_radius[:,0],yerr=arr_ml_radius[:,2])


    pl.show()
    import pdb; pdb.set_trace()



def main():


    valid_actions = ['figure_fields','figure_model','figure_contours','table_individual','figures_individual' , 'figure_vary_bf']

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )

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