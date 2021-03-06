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


logger = logging.getLogger("filam..fit") 
logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False

cospars = cosmology.cosmoparams()

prob_z = None

dtype_stats = {'names' : ['id','kappa0_signif', 'kappa0_map', 'kappa0_err_hi', 'kappa0_err_lo', 'radius_map',    'radius_err_hi', 'radius_err_lo', 'chi2_red_null', 'chi2_red_max',  'chi2_red_D', 'chi2_red_LRT' , 'chi2_null', 'chi2_max', 'chi2_D' , 'chi2_LRT' ,'sigma_g' ] , 
        'formats' : ['i8'] + ['f8']*16 }



def add_model_selection():

    filename_shears = config['filename_shears']                                  # args.filename_shears 
    filename_pairs = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')
    filename_grid = '%s/results.grid.%s.pp2' % (args.results_dir,name_data)
    grid_pickle = tabletools.loadPickle(filename_grid)
    filename_grid_const = 'results_const/results.grid.%s.pp2' % (name_data)
    grid_pickle_const = tabletools.loadPickle(filename_grid_const)


    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)
    if os.path.isfile('classification.txt'):
        classification = np.loadtxt('classification.txt',dtype='i4')[:,1]
        pairs = tabletools.ensureColumn(rec=pairs,arr=classification,name='eyeball_class',dtype='i4')

    pairs = tabletools.ensureColumn(rec=pairs,name='BF1',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='BF2',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='MLRT1',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='MLRT2',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='ML_kappa0',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='ML_radius',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='sigma_null',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='ML_kappa0_errhi',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='ML_kappa0_errlo',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='ML_radius_errhi',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='ML_radius_errlo',dtype='f4')
    pairs = tabletools.ensureColumn(rec=pairs,name='eyeball_class',dtype='f4')

    for ic in range(n_pairs):

        filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir, ic, ic+1, name_data)
        filename_pickle_cons = '%s/results.prob.%04d.%04d.%s.pp2'  % ('results_const', ic, ic+1, name_data)
        try:
            log_like = tabletools.loadPickle(filename_pickle,log=0)
            log_like_const = tabletools.loadPickle(filename_pickle_cons,log=0)
        except:
            logger.debug('missing %s' % filename_pickle)
            continue

        normalisation_const = log_like.max()
        prob_like = np.exp(log_like - normalisation_const)
        prob_like_2D = np.sum(prob_like,axis=(2,3))
        log_prob_2D = np.log(prob_like_2D)
        select= np.isinf(log_prob_2D) | np.isnan(log_prob_2D)
        min_val = log_prob_2D[~select].min()
        log_prob_2D[select] = min_val
        n_halo_grid  = log_like.shape[2]*log_like.shape[3]

        grid_kappa0 = grid_pickle['grid_kappa0'][:,:,0,0]
        grid_radius = grid_pickle['grid_radius'][:,:,0,0]
        vec_h1M200 = grid_pickle['grid_h1M200'][0,0,:,0]
        vec_h2M200 = grid_pickle['grid_h2M200'][0,0,0,:]
        vec_kappa0 = grid_kappa0[:,0]
        vec_radius = grid_radius[0,:]
       
        n_upsample = 20
        vec_kappa0_hires = np.linspace(min(grid_kappa0[:,0]),max(grid_kappa0[:,0]),len(grid_kappa0[:,0])*n_upsample)
        vec_radius_hires = np.linspace(min(grid_radius[0,:]),max(grid_radius[0,:]),len(grid_radius[0,:])*n_upsample)
        grid_kappa0_hires, grid_radius_hires = np.meshgrid(vec_kappa0_hires,vec_radius_hires,indexing='ij')
        import scipy.interpolate
        func_interp = scipy.interpolate.interp2d(vec_kappa0,vec_radius,log_prob_2D, kind='cubic')
        log_prob_2D_hires = func_interp(vec_kappa0_hires,vec_radius_hires)
        # prob_2D_hires = np.exp(log_prob_2D_hires-log_prob_2D_hires.max())
        prob_2D_hires = np.exp(log_prob_2D_hires)

        # pl.figure()
        # pl.pcolormesh(grid_kappa0_hires,grid_radius_hires,np.exp(log_prob_2D_hires-log_prob_2D_hires.max()))
        # pl.show()

        # constant model
        vec_kappa0_const = grid_pickle_const['grid_kappa0'][:,0,0,0] 
        vec_h1M200_const = grid_pickle_const['grid_h1M200'][0,0,:,0] 
        vec_h2M200_const = grid_pickle_const['grid_h2M200'][0,0,0,:] 
        log_like_const = np.squeeze(log_like_const,1)
        prob_like_const = np.exp(log_like_const - normalisation_const)
        prob_like_const_1D = np.sum(prob_like_const,axis=(1,2))
        log_like_const_1D = np.log(prob_like_const_1D)
        log_like_const_1D[np.isinf(log_like_const_1D)]=log_like_const_1D[~np.isinf(log_like_const_1D)].min()
        vec_kappa0_const_hires = np.linspace(vec_kappa0_const.min(),vec_kappa0_const.max(),len(vec_kappa0_const)*n_upsample)
        func_interp = scipy.interpolate.interp1d(np.linspace(0,1,len(prob_like_const_1D)),log_like_const_1D,'cubic')
        log_like_const_1D = func_interp(np.linspace(0,1,len(prob_like_const_1D)*n_upsample))
        # prob_like_const_1D = np.exp(log_like_const_1D-log_prob_2D_hires.max())
        prob_like_const_1D = np.exp(log_like_const_1D)
        n_halo_grid_const  = log_like_const.shape[1]*log_like_const.shape[2]

        max_density_prior = 500
        max_kappa0_prior = 0.7
        max_radius_prior = 3.0
        rho_crit = 2.77501358101e+11
        int_DS = 1e14*(grid_kappa0_hires * ( grid_radius_hires * np.pi ) /2.) 
        d_rs = (int_DS / rho_crit) 
        # select = (d_rs < max_density_prior) * (grid_kappa0_hires>1e-2) * (grid_radius_hires < max_radius_prior)
        select = (grid_kappa0_hires>1e-2) * (grid_radius_hires < max_radius_prior) * (grid_kappa0_hires < max_kappa0_prior)
        prob_2D_use = prob_2D_hires[select]
        # pl.figure()
        # pl.contour(grid_kappa0_hires,grid_radius_hires,d_rs,levels=[max_density_prior,0])
        # pl.show()

        max_const_prior = 100
        select = vec_kappa0_const_hires < max_const_prior
        prob_like_const_1D_use = prob_like_const_1D[select]

        d1=prob_2D_use.size  * n_halo_grid
        d2=prob_2D_hires[0,0].size * n_halo_grid
        d3=prob_like_const_1D_use.size * n_halo_grid_const
        warnings.warn('f1=%d f2=%d f3=%d' % (d1,d2,d3)) 
        bf1 = (   np.sum(prob_2D_use) / d1 )  / ( prob_2D_hires[0,0]             / d2  )
        bf2 = (   np.sum(prob_2D_use) / d1 )  / ( np.sum(prob_like_const_1D_use) / d3    )
        if np.isnan(bf1): 
            print 'bf1 is nan'
            bf1 = 0.

        if np.isnan(bf2): 
            bf2 = 0.
            print 'bf2 is nan'

        MLRT1 = maximum_likelihood_ratio_test(log_like[1:,0:5,:,:],log_like[0,0,:,:],2)
        MLRT2 = maximum_likelihood_ratio_test(log_like[1:,0:5,:,:],log_like_const,1)

        pairs['BF1'][ic] = bf1
        pairs['BF2'][ic] = bf2
        pairs['MLRT1'][ic] = MLRT1
        pairs['MLRT2'][ic] = MLRT2

        status_str='ok'
        if ((pairs['BF1'][ic]>1) & (pairs['BF2'][ic]>1) & (pairs['eyeball_class'][ic] == 2)): status_str = '!! FP !!'
        if (((pairs['BF1'][ic]<1) | (pairs['BF2'][ic]<1)) & (pairs['eyeball_class'][ic] == 1)): status_str = 'miss'

        print '% 4d ih1=% 5d ih2=% 5d m200_h1=%2.2f m200_h2=%2.2f BF1=%12.3f \t BF2=%12.3f \t\tMLRT1=%12.3f\t\tMLRT2=%12.3f\t\tclass=%d %s' % (ic, pairs[ic]['ih1'] , pairs[ic]['ih2'], pairs[ic]['m200_h1_fit'], pairs[ic]['m200_h2_fit'],bf1,bf2,MLRT1,MLRT2,pairs['eyeball_class'][ic],status_str)
        # import pdb; pdb.set_trace()

        max_model = np.unravel_index(prob_2D_hires.argmax(), prob_2D_hires.shape)
        ML_kappa0 = grid_kappa0_hires[max_model]
        ML_radius = grid_radius_hires[max_model]
        pairs['ML_kappa0'][ic] = ML_kappa0
        pairs['ML_radius'][ic] = ML_radius
        import scipy.special
        pairs['sigma_null'][ic] = scipy.special.erfinv(1.-MLRT1)*np.sqrt(2.)
        max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_kappa0_hires[:,max_model[1]],prob_2D_hires[:,max_model[1]])
        pairs['ML_kappa0_errhi'][ic] =err_hi
        pairs['ML_kappa0_errlo'][ic] =err_lo

        max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_radius_hires[max_model[0],:],prob_2D_hires[max_model[0],:])
        pairs['ML_radius_errhi'][ic] =err_hi
        pairs['ML_radius_errlo'][ic] =err_lo


        # if ic == 73:
        #     import pdb; pdb.set_trace()






    # pl.figure();pl.scatter(pairs['BF1'],pairs['eyeball_class']); pl.xlim([0,100]); 
    # pl.figure();pl.scatter(pairs['BF2'],pairs['eyeball_class']); pl.xlim([0,100]); 
    # pl.figure();pl.scatter(pairs['MLRT1'],pairs['eyeball_class']); pl.xlim([0,1]); 
    # pl.figure();pl.scatter(pairs['MLRT2'],pairs['eyeball_class']); pl.xlim([0,1]); 
    # pl.show()
    tabletools.saveTable(filename_pairs,pairs)

    print 'MLRT n_clean '        , sum( (pairs['MLRT1']<0.2) & (pairs['MLRT2']<0.2) & (pairs['eyeball_class']==1))
    print 'MLRT n_contaminating ', sum( (pairs['MLRT1']<0.2) & (pairs['MLRT2']<0.2) & (pairs['eyeball_class']!=1))
    print 'BF missed '         ,   sum( ((pairs['BF1']<1) | (pairs['BF2']<1)) & ( (pairs['eyeball_class']==1) | (pairs['eyeball_class']==1) )  )
    print 'BF n_clean '        ,   sum( (pairs['BF1']>1) & (pairs['BF2']>1) & ( (pairs['eyeball_class']==1) | (pairs['eyeball_class']==3) )  )
    print 'BF n_contaminating ',   sum( (pairs['BF1']>1) & (pairs['BF2']>1) & ( (pairs['eyeball_class']==2) | (pairs['eyeball_class']==0) )  )

    import pdb; pdb.set_trace()




def maximum_likelihood_ratio_test(log_like_model,log_like_null,ndof):

    max_null = np.unravel_index(log_like_null.argmax(), log_like_null.shape)
    max_model = np.unravel_index(log_like_model.argmax(), log_like_model.shape)
    chi2_null = log_like_null[max_null]
    chi2_model = log_like_model[max_model]
    D = 2*(chi2_model - chi2_null)
    ndof=2
    LRT_pval = 1. - scipy.stats.chi2.cdf(D, ndof)
    # print 'chi2_max_null=%6.2f chi2_max_model=%6.2f chi2_red_null=%6.6f chi2_red_model=%6.6f chi2_D_red=%6.2e chi2_LRT_red=%6.2e max_model=%s max_null=%s' % (chi2_max_null, chi2_max_model, chi2_red_null , chi2_red_model, chi2_D_red, chi2_LRT_red, str(max_model), str(max_null))

    return LRT_pval

def figure_fields():

    box_w1 = [29.5,39.5,-12,-3]
    box_w2 = [208,221,50.5,58.5]
    box_w3 = [329.5,336,-2,5.5]
    box_w4 = [131.5,137.5,-6.5,-0.5]

    halos = tabletools.loadTable(config['filename_halos'])
    filename_halos_cfhtlens = os.environ['HOME'] + '/data/CFHTLens/CFHTLens_DR10_LRG/BOSSDR10LRG.fits'
    bossdr10 = pyfits.getdata(filename_halos_cfhtlens)
    filename_cluscat = os.environ['HOME'] + '/data/CFHTLens/ClusterZ/clustersz.fits'
    pairs = tabletools.loadTable(config['filename_pairs'])
    halo1 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos1.fits'))
    halo2 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos2.fits'))
    cluscat = tabletools.loadTable(filename_cluscat)

    if 'm200_h1_fit' not in pairs.dtype.names:
        if 'm200' in halos.dtype.names:
            pairs = tabletools.appendColumn(rec=pairs,arr=np.log10(halo1['m200']),name='m200_h1_fit')
            pairs = tabletools.appendColumn(rec=pairs,arr=np.log10(halo2['m200']),name='m200_h2_fit')
            
    mass= (pairs['m200_h1_fit']+pairs['m200_h2_fit'])/2.
    # mass= halo1['m200']
    select = (mass < 16.5) * (10.**mass > 1e14)

    pairs=pairs[select]
    halo1=halo1[select]
    halo2=halo2[select]

    # print 'using %d pairs' % len(halo2)

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as pl
    m = Basemap(projection='ortho',lat_0=-35,lon_0=-10,resolution='c')
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    m.drawmapboundary()

    x1,y1 = m(pairs['ra1'],pairs['dec1'])
    x2,y2 = m(pairs['ra2'],pairs['dec2'])
    x3,y3 = m(halos['ra'],halos['dec'])

    m.scatter(x1,y1, pairs['m200_h1_fit'] , pairs['z'] , marker = 'o') #
    m.scatter(x2,y2, pairs['m200_h2_fit'] , pairs['z'] , marker = 'o') #
    m.scatter(x3,y3, halos['m200']     , halos['z'] , marker = 'o') #

    for i in range(len(pairs)):
        # m.scatter([x1[i],x2[i]],[y1[i],y2[i]] , c=table_halo2['z'][i] , cmap=pl.matplotlib.cm.jet)
        m.plot([x1[i],x2[i]],[y1[i],y2[i]])



    # pl.figure()
    # pl.scatter(halos['ra'],halos['dec'],s=2,c=halos['z'])
    # pl.scatter(pairs['ra1'],pairs['dec1'], 50 , c=halo1['z'] , marker = 'o' ) #
    # pl.scatter(pairs['ra2'],pairs['dec2'], 50 , c=halo2['z'] , marker = 'o' ) #      
    # for i in range(len(pairs)): pl.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c='r')
    # pl.xlabel('RA')
    # pl.ylabel('Dec')
    # # pl.colorbar()

    filename_fig = 'filament_map.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' , filename_fig)
    pl.show()
    import pdb; pdb.set_trace()

def figures_proximity():

    pairs_colors = ['r','g','b','y','y']
    clean_colors = ['k','w']


    halos = tabletools.loadTable(config['filename_halos'])
    filename_halos_cfhtlens = os.environ['HOME'] + '/data/CFHTLens/CFHTLens_DR10_LRG/BOSSDR10LRG.fits'
    filename_cluscat = os.environ['HOME'] + '/data/CFHTLens/Clusters/Clusters.fits'
    filename_fields =  os.environ['HOME'] + '/data/CFHTLens/field_catalog.fits'
    bossdr10 = pyfits.getdata(filename_halos_cfhtlens)
    pairs = tabletools.loadTable(config['filename_pairs'])
    halo1 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos1.fits'))
    halo2 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos2.fits'))
    cluscat = tabletools.loadTable(filename_cluscat)
    fieldscat = tabletools.loadTable(filename_fields)
    select = (cluscat['sig'] > 5) * (cluscat['z'] < 2)
    cluscat = cluscat[select]
    cluscat.dtype.names = [n.lower() for n in cluscat.dtype.names]

    select = halos['m200_fit'] > 5e13
    halos = halos[select]

    minz , maxz = 0.2 , 0.8
    def match_ne(vp):
        matched = [(vp['ra1']*10000).astype(np.int32),(vp['ra2']*10000).astype(np.int32)] in (halos['ra'][ne]*10000).astype(np.int32)
        return matched

    def plot_field(ax,box_w):

        import scipy.spatial
        import pylab as pl

        # ax.scatter(cluscat['ra'],cluscat['dec'],s=50,c=cluscat['z'],marker='d', vmin=minz, vmax=maxz)
        ax.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s', vmin=minz, vmax=maxz)

        for i in range(len(pairs)): 
            # print match_ne(pairs[i])
            ax.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c=pairs_colors[pairs['eyeball_class'][i]],lw=9)
            ax.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c=clean_colors[pairs['analysis'][i]],lw=5)
            ax.text((pairs['ra1'][i]+pairs['ra2'][i])/2.,(pairs['dec1'][i]+pairs['dec2'][i])/2.,'%d %2.2f'%(pairs[i]['ipair'],pairs[i]['z']),fontsize=10)
            # radius=cosmology.get_angular_separation(pairs['ra1'][i],pairs['dec1'][i],pairs['ra2'][i],pairs['dec2'][i])/2.
            # circle=pl.Circle((pairs['ra_mid'][i],pairs['dec_mid'][i]),radius,color='y',fill=False)
            # ax.add_artist(circle)
        ax.scatter(pairs['ra1'],pairs['dec1'], 100 , c=halo1['z'] , marker = 'o' , vmin=minz, vmax=maxz) #
        ax.scatter(pairs['ra2'],pairs['dec2'], 100 , c=halo2['z'] , marker = 'o' , vmin=minz, vmax=maxz) #
        for f in range(len(fieldscat)):
            x1=fieldscat[f]['ra_min']
            x2=fieldscat[f]['de_min']
            l1=fieldscat[f]['ra_max'] - fieldscat[f]['ra_min']
            l2=fieldscat[f]['de_max'] - fieldscat[f]['de_min']
            if fieldscat[f]['bad']==1:
                rect = pl.matplotlib.patches.Rectangle((x1,x2), l1, l2, facecolor='red', edgecolor='Black', linewidth=1.0 , alpha = 0.5)
            else:
                rect = pl.matplotlib.patches.Rectangle((x1,x2), l1, l2, facecolor=None, edgecolor='Black', linewidth=1.0 , fill=None)
            ax.add_patch(rect)
        for h in range(len(cluscat)):
            r200_mpc = cluscat['r200'][h]
            # r200_mpc = 1
            z = cluscat['z'][h]
            r200_deg = r200_mpc/cosmology.get_ang_diam_dist(z) * 180 / np.pi
            # circle=pl.Circle((cluscat['ra'][h],cluscat['dec'][h]),r200_deg,color='b',fill=False)
            # ax.add_artist(circle)
            # ax.text(cluscat['ra'][h],cluscat['dec'][h],'%2.2f %1.2f'%(cluscat['z'][h],cluscat['m200'][h]),fontsize=10)
        for i in range(len(halos)):
            ax.text(halos['ra'][i],halos['dec'][i],' %d %2.1e %2.4f'% (i,halos['m200_fit'][i],halos['z'][i]),fontsize=10)

        # pl.plot([halos['ra'][ne[:,0]],halos['ra'][ne[:,1]]],[halos['dec'][ne[:,0]],halos['dec'][ne[:,1]]],c='y',lw=1)



        ax.set_xlim(box_w[0],box_w[1])
        ax.set_ylim(box_w[2],box_w[3])


    for ip,vp in enumerate(pairs):

        # if vp['analysis'] == 0:
        #     continue

        min_ra = min([vp['ra1'],vp['ra2']])   -1
        max_ra = max([vp['ra1'],vp['ra2']])   +1
        min_de = min([vp['dec1'],vp['dec2']]) -1
        max_de = max([vp['dec1'],vp['dec2']]) +1
        box_w = [min_ra,max_ra,min_de,max_de]
        pl.figure(figsize=(10,10))
        plot_field(pl.gca(),box_w)
        Dtot=np.sqrt(vp['Dxy']**2+vp['Dlos']**2)
        titlestr = '%d analysis=%d class=%d Dlos=%2.2f Dxy=%2.2f Dtot=%2.2f m1=%1.1e m2=%1.1e' % (ip,vp['analysis'],vp['eyeball_class'],vp['Dlos'],vp['Dxy'],Dtot,halo1[ip]['m200_fit'],halo2[ip]['m200_fit'])
        pl.title(titlestr)
        filename_fig = 'figs/proximity.%04d.png' % ip
        pl.savefig(filename_fig,dpi=150)
        pl.close()
        logger.info('saved %s',filename_fig)




def figure_fields_cfhtlens():

    pairs_colors = ['r','g','b','y','y']
    clean_colors = ['k','w']

    box_w1 = [29.5,39.5,-12,-3]
    box_w2 = [208,221,50.5,58.5]
    box_w3 = [329.5,336,-2,5.5]
    box_w4 = [131.5,137.5,-6.5,-0.5]

    halos = tabletools.loadTable(config['filename_halos'])
    filename_halos_cfhtlens = os.environ['HOME'] + '/data/CFHTLens/CFHTLens_DR10_LRG/BOSSDR10LRG.fits'
    filename_cluscat = os.environ['HOME'] + '/data/CFHTLens/CFHTLS_wide_clusters_Durret2011/wide.fits'
    # filename_cluscat = os.environ['HOME'] + '/data/CFHTLens/Clusters/Clusters.fits'
    filename_fields =  os.environ['HOME'] + '/data/CFHTLens/field_catalog.fits'
    bossdr10 = pyfits.getdata(filename_halos_cfhtlens)
    pairs = tabletools.loadTable(config['filename_pairs'])
    halo1 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos1.fits'))
    halo2 = tabletools.loadTable(config['filename_pairs'].replace('.fits','.halos2.fits'))
    cluscat = tabletools.loadTable(filename_cluscat)
    fieldscat = tabletools.loadTable(filename_fields)
    # select = (cluscat['m200'] > np.log10(4e13)) * (cluscat['z'] < 2)
    # cluscat = cluscat[select]
    cluscat.dtype.names = [n.lower() for n in cluscat.dtype.names]

    select = halos['m200_fit'] > 1e12
    halos = halos[select]

    select = pairs['analysis'] == 1
    pairs=pairs[select]
    halo1=halo1[select]
    halo2=halo2[select]

    # select = (halo1['m200_fit'] > 1e14) & (halo2['m200_fit'] > 1e14)
    # pairs=pairs[select]
        
    # n_pairs_used = len(pairs)
    print 'using %d pairs' % sum( pairs['analysis']==1)
    print 'eyeball_class 0', sum( (pairs['eyeball_class']==0))
    print 'eyeball_class 1', sum( (pairs['eyeball_class']==1))
    print 'eyeball_class 2', sum( (pairs['eyeball_class']==2))

    try:
        import matplotlib
        import matplotlib.gridspec as gridspec
    except:
        logger.error('gridspec not found - no plot today')
        return None

    import matplotlib.gridspec as gridspec
    fig = pl.figure(1)
    fig.clf()
    gs = gridspec.GridSpec( 2, 2, width_ratios=[1,1], height_ratios=[2,1], wspace=0.25, hspace=0.25)
    ax1 = fig.add_subplot(gs[0]) # 7x10
    ax2 = fig.add_subplot(gs[1]) # 7x12
    ax3 = fig.add_subplot(gs[2]) # 7x6
    ax4 = fig.add_subplot(gs[3]) # 2x5

    fig.text(0.5, 0.04, 'RA', ha='center', va='center')
    fig.text(0.06, 0.5, 'Dec', ha='center', va='center', rotation='vertical')

    minz=0.2
    maxz=0.8
    def plot_field(ax,box_w):

        # ax.scatter(cluscat['ra'],cluscat['de'],s=50,c=cluscat['zphot'],marker='d', vmin=minz, vmax=maxz)
        ax.scatter(halos['ra'],halos['dec'],s=50,c=halos['z'],marker='s', vmin=minz, vmax=maxz)
        for i in range(len(pairs)): 
            ax.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c=pairs_colors[pairs['eyeball_class'][i]],lw=5)
            ax.plot([pairs['ra1'][i],pairs['ra2'][i]],[pairs['dec1'][i],pairs['dec2'][i]],c=clean_colors[pairs['analysis'][i]],lw=2)
            ax.text((pairs['ra1'][i]+pairs['ra2'][i])/2.,(pairs['dec1'][i]+pairs['dec2'][i])/2.,'%d %2.2f'%(pairs[i]['ipair'],pairs[i]['z']),fontsize=10)
            # radius=cosmology.get_angular_separation(pairs['ra1'][i],pairs['dec1'][i],pairs['ra2'][i],pairs['dec2'][i])/2.
            # circle=pl.Circle((pairs['ra_mid'][i],pairs['dec_mid'][i]),radius,color='y',fill=False)
            # ax.add_artist(circle)
        ax.scatter(pairs['ra1'],pairs['dec1'], 100 , c=halo1['z'] , marker = 'o' , vmin=minz, vmax=maxz) #
        ax.scatter(pairs['ra2'],pairs['dec2'], 100 , c=halo2['z'] , marker = 'o' , vmin=minz, vmax=maxz) #
        for f in range(len(fieldscat)):
            x1=fieldscat[f]['ra_min']
            x2=fieldscat[f]['de_min']
            l1=fieldscat[f]['ra_max'] - fieldscat[f]['ra_min']
            l2=fieldscat[f]['de_max'] - fieldscat[f]['de_min']
            if fieldscat[f]['bad']==1:
                rect = matplotlib.patches.Rectangle((x1,x2), l1, l2, facecolor='red', edgecolor='Black', linewidth=1.0 , alpha = 0.5)
            else:
                rect = matplotlib.patches.Rectangle((x1,x2), l1, l2, facecolor=None, edgecolor='Black', linewidth=1.0 , fill=None)
            ax.add_patch(rect)
        for h in range(len(cluscat)):
            pass
            # r200_mpc = cluscat['r200'][h]
            # r200_mpc = 1
            # z = cluscat['z'][h]
            # r200_deg = r200_mpc/cosmology.get_ang_diam_dist(z) * 180 / np.pi
            # circle=pl.Circle((cluscat['ra'][h],cluscat['dec'][h]),r200_deg,color='b',fill=False)
            # ax.add_artist(circle)
            # ax.text(cluscat['ra'][h],cluscat['dec'][h],'%2.2f'%(cluscat['z'][h]),fontsize=10)
        for h in range(len(halos)):
            if halos['m200_sig'][h] < 1: continue
            r200_mpc = 10
            # z = halos['z'][h]
            # r200_deg = r200_mpc/cosmology.get_ang_diam_dist(z) * 180 / np.pi
            # circle=pl.Circle((halos['ra'][h],halos['dec'][h]),r200_deg,color='b',fill=False)
            # ax.add_artist(circle)

            pass
            # ax.text(halos['ra'][i],halos['dec'][i],' %d %2.1e'% (i,halos['m200_fit'][i]),fontsize=10)

        ax.set_xlim(box_w[0],box_w[1])
        ax.set_ylim(box_w[2],box_w[3])
        ax.invert_xaxis()


    plot_field(ax1,box_w1)
    plot_field(ax2,box_w2)
    plot_field(ax3,box_w3)
    plot_field(ax4,box_w4)

    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])
    # fig.colorbar(cax,cax=cbar_ax)
    # fig.suptitle('%d pairs - class %d' % (n_pairs_used, classif))
    filename_fig = 'filament_map.png'
    # pl.savefig(filename_fig)
    # logger.info('saved %s' , filename_fig)
    fig.show()
    # fig.close()
    import pdb; pdb.set_trace()


def add_stats():

    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_halos']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halos = tabletools.loadTable(filename_halos)
    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)
    n_halos = len(halos)

    if not 'm200_h1_fit' in pairs.dtype.names:
        pairs = tabletools.appendColumn(arr=np.zeros(n_pairs),rec=pairs,name='m200_h1_fit',dtype='f4')
        pairs = tabletools.appendColumn(arr=np.zeros(n_pairs),rec=pairs,name='m200_h2_fit',dtype='f4')

    if not 'm200_fit' in halos.dtype.names:
        halos = tabletools.appendColumn(arr=np.zeros(n_halos),rec=halos,name='m200_fit',dtype='f4')

    filename_grid = '%s/results.grid.%s.pp2' % (args.results_dir,name_data)
    grid_dict = tabletools.loadPickle(filename_grid)

    id_file_first = args.first_result_file
    id_file_last = id_file_first + args.n_results_files
    
    n_missing=0
    for nf in range(id_file_first,id_file_last):

            filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir,nf, nf+1 , name_data)
            try:
                res = tabletools.loadPickle(filename_pickle,log=1)
            except:
                logger.debug('missing %s' % filename_pickle)
                n_missing +=1
                continue
            if len(res) == 0:
                logger.debug('empty %s' % filename_pickle)
                n_missing +=1
                continue

            prod_pdf = mathstools.normalise(res)
            grid_h1M200 = grid_dict['grid_h1M200'][0,0,:,0]
            grid_h2M200 = grid_dict['grid_h2M200'][0,0,0,:]

            prod_pdf_h1M200 = np.sum(prod_pdf,axis=(0,1,3))
            prod_pdf_h2M200 = np.sum(prod_pdf,axis=(0,1,2))

            # ml_mass_h1 = grid_h1M200[np.argmax(prod_pdf_h1M200)]
            # ml_mass_h2 = grid_h2M200[np.argmax(prod_pdf_h2M200)]
            ml_mass_h1 = grid_dict['grid_h1M200'].flatten()[np.argmax(prod_pdf)]
            ml_mass_h2 = grid_dict['grid_h2M200'].flatten()[np.argmax(prod_pdf)]

            pairs['m200_h1_fit'][nf] = ml_mass_h1
            pairs['m200_h2_fit'][nf] = ml_mass_h2

            ih1 = pairs['ih1'][nf]
            ih2 = pairs['ih2'][nf]
            halos['m200_fit'][ih1] = ml_mass_h1
            halos['m200_fit'][ih2] = ml_mass_h2

            logger.info('%4d m200_h1_fit=%2.4e m200_h2_fit=%2.4e %d %d' % (nf,ml_mass_h1,ml_mass_h2,ih1,ih2) )

            # pl.plot(grid_h1M200,prod_pdf_h1M200,'r')
            # pl.plot(grid_h2M200,prod_pdf_h2M200,'g')
            # pl.axvline(ml_mass_h1,c='r')
            # pl.axvline(ml_mass_h2,c='g')
            # pl.axvline(halo1['m200'][nf],c='r',marker='o')
            # pl.axvline(halo2['m200'][nf],c='g',marker='o')          
            # pl.show()

    tabletools.saveTable(filename_pairs,pairs)
    tabletools.saveTable(filename_halos,halos)






def get_prob_prod_mcmc(ids):

    n_per_file = 10
    id_file_first = args.first_result_file
    id_file_last = id_file_first + args.n_results_files
    n_params = 4
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','')

    prob_prod = []
    for ip in range(n_params): prob_prod.append(np.zeros(config['n_grid']))
    prob_kappa0_radius = np.zeros([config['n_grid_2D'],config['n_grid_2D']])
    
    n_missing=0
    n_usable_results=0
    ia=0

    filename_grid = 'results/results.grid.%s.pp2' % name_data
    grid_pickle = tabletools.loadPickle(filename_grid)
    grid = grid_pickle['list_params_marg']
    grid2D = grid_pickle['kappa0_radius_grid']

    grid2D_kappa0 = np.reshape(grid_pickle['kappa0_radius_grid'][:,0],[config['n_grid_2D'],config['n_grid_2D']])
    grid2D_radius = np.reshape(grid_pickle['kappa0_radius_grid'][:,1],[config['n_grid_2D'],config['n_grid_2D']])

    for nf in range(id_file_first,id_file_last):

        id_start = nf*n_per_file
        id_end = (nf+1)*n_per_file
        current_ids = range(id_start,id_end)

        if len(list(set(ids) & set(current_ids))) < 1:

            ia+=n_per_file
            logger.debug('no requested results between %d and %d' , id_start, id_end)
            continue

        else:

            filename_pickle = 'results/results.chain.%04d.%04d.%s.pp2'  % (nf*n_per_file, (nf+1)*n_per_file , name_data)
            try:
                results_pickle = tabletools.loadPickle(filename_pickle,log=1)
            except:
                logger.info('missing %s' % filename_pickle)
                n_missing +=1
                ia+=n_per_file
                continue

            for ni in range(n_per_file):

                if ia in ids:

                    # marginals in each parameter
                    for ip in range(n_params):

                        # turn this off
                        if ia>0: break

                        prob = results_pickle[ni]['list_prob_marg'][ip]
                        nans = np.nonzero(np.isnan(prob))[0]
                        if len(nans) > 0:
                            logger.info('%d %s param=%d n_nans=%d', ni, filename_pickle , ip, len(np.nonzero(np.isnan(prob))[0]) )
                        prob[prob<1e-20]=1e-20
                        logprob = prob
                        # nans = np.nonzero(np.isnan( logprob))[0]
                        # infs = np.nonzero(np.isinf( logprob))[0]
                        # zeros = np.nonzero( logprob == 0.0)[0]

                        prod1D_pdf , prod1D_log_pdf , _ , _ = mathstools.get_normalisation(logprob)
                        try:
                            prob_prod[ip] += prod1D_log_pdf
                        except Exception,errmsg:
                            print errmsg
                            import pdb; pdb.set_trace()

                    # marginal kappa-radius
                    logprob2D = np.log(results_pickle[ni]['prob_kappa0_radius'])
                    # logprob2D[logprob2D<1e-20]=1e-20

                    # prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob2D)  
                    # nans = np.nonzero(np.isnan(prod2D_pdf))[0]
                    # n_nans=len(nans)
                    # if n_nans > 0: logger.info('n_nans=%d',n_nans)
                    # prod2D_log_pdf = np.reshape(prod2D_log_pdf,[config['n_grid_2D'],config['n_grid_2D']])
                    
                    prod2D_log_pdf = np.reshape(logprob2D,[config['n_grid_2D'],config['n_grid_2D']])
                    prob_kappa0_radius += prod2D_log_pdf

                    # plot_prob_all, _, _,_ = mathstools.get_normalisation(prob_kappa0_radius)  
                    # plot_prob_this, _, _,_ = mathstools.get_normalisation(prod2D_log_pdf)                    
                    # pl.figure(figsize=(10,5))
                    # pl.subplot(1,2,1)
                    # pl.pcolormesh(grid2D_kappa0 , grid2D_radius , plot_prob_all); pl.colorbar()
                    # pl.xlim([-0.2,0.2])
                    # pl.ylim([-10,10])
                    # pl.subplot(1,2,2)
                    # pl.pcolormesh(grid2D_kappa0 , grid2D_radius , plot_prob_this); pl.colorbar()
                    # pl.xlim([-0.2,0.2])
                    # pl.ylim([-10,10])
                    # pl.show()

                    n_usable_results+=1
                ia+=1
                
            logger.info('%4d %s n_usable_results=%d' , nf , filename_pickle , n_usable_results)

    for ip in range(n_params):
        prob1D_pdf , prod1D_log_pdf , _ , _ = mathstools.get_normalisation(prob_prod[ip])  
        prob_prod[ip] = prob1D_pdf
    prob1D_pdf = prob_prod

    prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(prob_kappa0_radius)  

        
    return prob1D_pdf, grid, prod2D_pdf, grid2D_kappa0, grid2D_radius, n_usable_results

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def get_prob_prod_gridsearch(ids):

    id_file_first = args.first_result_file
    id_file_last = id_file_first + args.n_results_files
    n_params = 4
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')
  
    n_missing=0
    n_usable_results=0

    filename_grid = '%s/results.grid.%s.pp2' % (args.results_dir,name_data)
    grid_pickle = tabletools.loadPickle(filename_grid)

    res_all = None

    for nf in range(id_file_first,id_file_last):

        if nf in ids:

            filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir,nf, nf+1 , name_data)
            try:
                res = tabletools.loadPickle(filename_pickle,log=1)
            except:
                logger.debug('missing %s' % filename_pickle)
                n_missing +=1
                continue
            if len(res) == 0:
                logger.debug('empty %s' % filename_pickle)
                n_missing +=1
                continue

            if (res_all !=None):
                if (type(res)!=type(res_all)):
                    print 'something wrong with the array', res
                    continue
                if (res.shape != res_all.shape):
                    print 'something wrong with the shape', res
                    continue

            res_all = res if res_all == None else res_all+res

            n_usable_results+=1
            logger.debug('%4d %s n_usable_results=%d' , nf , filename_pickle , n_usable_results)
            if nf % 100 == 0 : logger.info('%4d n_usable_results=%d' , nf , n_usable_results)


    prod_pdf = mathstools.normalise(res_all)   

    return prod_pdf, grid_pickle, n_usable_results
    
def apply_prior(log_prob,grid_h1M200,grid_h2M200,ids=None):


    filename_prior = config['filename_halos'].replace('.fits','.prior.pp2')  
    prior_dict = tabletools.loadPickle(filename_prior,log=0)
    pairs = tabletools.loadTable(config['filename_pairs'],log=0)
    halos_use = np.concatenate([pairs[ids]['ih1'],pairs[ids]['ih2']])

    if 'use_random_halos' in config:
        warnings.warn('using random halos')
        pairs_main = tabletools.loadTable('pairs_cfhtlens_lrgs.fits',log=0)
        pairs_main = pairs_main[pairs_main['analysis']==1]
        halos_use = np.concatenate([pairs_main['ih1'],pairs_main['ih2']])

    prior_ids = prior_dict['halos_like'][halos_use]
    norm1 = prior_ids-np.kron( np.ones([1,prior_ids.shape[1]]) , prior_ids.max(axis=1)[:,None] )
    norm2 = np.exp(norm1)/np.kron( np.ones([1,prior_ids.shape[1]]) , np.sum(np.exp(norm1),axis=1)[:,None] )
    norm3 = np.sum(norm2,axis=0)
    import scipy.interpolate
    x=prior_dict['grid_M200']/1e14

    if args.halo_removal == 'prior':  y = np.log(norm3/np.sum(norm3))      
    if args.halo_removal == 'flat':   y=x*0-1

    min_m200_h1 = 1
    min_m200_h2 = 1
    max_m200_h1 = 10
    max_m200_h2 = 10
    y1 = y.copy()
    y2 = y.copy()
    y1[x<min_m200_h1] = y.min()*1000
    y2[x<min_m200_h2] = y.min()*1000
    y1[x>max_m200_h1] = y.min()*1000
    y2[x>max_m200_h2] = y.min()*1000

    # pl.plot(x,np.exp(y2-y2.max())); pl.show()

    fun=scipy.interpolate.interp1d(x,y1)
    log_prior1=np.reshape(fun(grid_h1M200.flatten()),grid_h1M200.shape)
    fun=scipy.interpolate.interp1d(x,y2)
    log_prior2=np.reshape(fun(grid_h2M200.flatten()),grid_h2M200.shape)

    warnings.warn('applying prior')

    log_prob = log_prob+log_prior1+log_prior2

    return log_prob

def get_prob_prod_gridsearch_2D(ids,plots=False,hires=True,hires_marg=False,normalisation_const=None):

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    halos = tabletools.loadTable(config['filename_halos'])
    
    import scipy.interpolate
    n_per_file = 1
    id_file_first = args.first
    if args.num==-1:
        id_file_last = max(ids)+1
    elif id_file_first+args.num>max(ids):
        id_file_last = max(ids)+1
    else:
        id_file_last=id_file_first+args.num

    n_params = 4
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')

    # initialise lists for 1D marginals
    list_pdf_prod_1D = []
    list_pdf_grid_1D = []
   
    n_missing=0
    n_usable_results=0

    filename_grid = '%s/results.grid.%s.pp2' % (args.results_dir,name_data)
    grid_pickle = tabletools.loadPickle(filename_grid)
    grid_kappa0 = grid_pickle['grid_kappa0'][:,:,0,0]
    grid_radius = grid_pickle['grid_radius'][:,:,0,0]
    vec_kappa0 = grid_kappa0[:,0]
    vec_radius = grid_radius[0,:]
    logprob_kappa0_radius = np.zeros([ len(grid_kappa0[:,0]) , len(grid_radius[0,:]) ])+66
    grid2D_dict = { 'grid_kappa0'  : grid_kappa0 , 'grid_radius' : grid_radius}  
    list_ids_used = []

    normalisation_factor=-10000

    if hires:    
        n_upsample = 40
        vec_kappa0_hires = np.linspace(min(grid_kappa0[:,0]),max(grid_kappa0[:,0]),len(grid_kappa0[:,0])*n_upsample)
        # vec_radius_hires = np.linspace(min(grid_radius[0,:]),max(grid_radius[0,:]),len(grid_radius[0,:])*n_upsample)
        vec_radius_hires = np.linspace(min(grid_radius[0,:]),max(grid_radius[0,:]),len(grid_radius[0,:])*n_upsample)
        grid_kappa0_hires, grid_radius_hires = np.meshgrid(vec_kappa0_hires,vec_radius_hires,indexing='ij')
        logprob_kappa0_radius_hires = np.zeros([ len(vec_kappa0_hires) , len(vec_radius_hires) ])+66
        logprob_kappa0_hires = np.zeros([ len(vec_kappa0_hires) ])+66

    total_normalisation = 0
    for nf in range(id_file_first,id_file_last):

        if nf in ids:

            # filename_pickle = 'results_local2scratch/results.prob.%04d.%04d.%s.pp2'  % (nf*n_per_file, (nf+1)*n_per_file , name_data)
            filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir,nf*n_per_file, (nf+1)*n_per_file , name_data)
            try:
                results_pickle = tabletools.loadPickle(filename_pickle,log=1)
            except:
                logger.info('missing %s' % filename_pickle)
                n_missing +=1
                continue
            if len(results_pickle) == 0:
                logger.info('empty %s' % filename_pickle)
                n_missing +=1
                continue

            try:                
                log_prob = results_pickle['log_post']
                log_prob_2D = results_pickle['log_post_2D']
            except:
                log_prob = results_pickle
        
            fudge = 1.
            log_prob *= fudge
            warnings.warn('adjusting noise_std by %2.2f'%fudge)

            # marginal kappa-radius
            # log_prob = results_pickle*214.524/2.577
            warnings.warn('log_prob.shape %s' % str(log_prob.shape))

            m200_imin = 0
            m200_imax = log_prob.shape[3]
            grid_h1M200 = grid_pickle['grid_h1M200'][0,0,:,0]
            grid_h2M200 = grid_pickle['grid_h2M200'][0,0,0,:]
            log_prob = log_prob[:,:,m200_imin:m200_imax,m200_imin:m200_imax]
            warnings.warn('m200_min, m200_max %2.3f %2.3f' % (grid_h1M200[m200_imin],grid_h1M200[m200_imax-1]))
            warnings.warn('m200_min, m200_max %2.3f %2.3f' % (grid_h2M200[m200_imin],grid_h2M200[m200_imax-1]))
            if hires_marg:

                grid_h1M200_hires=np.linspace(grid_h1M200.min(),grid_h1M200.max(),len(grid_h1M200)*n_upsample)
                grid_h2M200_hires=np.linspace(grid_h2M200.min(),grid_h2M200.max(),len(grid_h2M200)*n_upsample)
                log_prob_2D = np.zeros_like(log_prob[:,:,0,0])

                for i1 in range(len(log_prob[:,0,0,0])):
                    for i2 in range(len(log_prob[0,:,0,0])):
                        m200_2D = log_prob[i1,i2,:,:]
                        func_interp = scipy.interpolate.interp2d(grid_h1M200,grid_h2M200,m200_2D, kind='cubic')
                        m200_2D_hires = func_interp(grid_h1M200_hires,grid_h2M200_hires)
                        normalisation_const=2000
                        m200_2D_hires_prob=np.exp(m200_2D_hires-log_prob.max())
                        log_prob_2D[i1,i2] = np.log(np.sum(m200_2D_hires_prob))
                        
                        if plots:
                            print grid_kappa0[i1,i2], grid_radius[i1,i2] , np.sum(mathstools.normalise(m200_2D_hires))/len(m200_2D_hires) , np.sum(mathstools.normalise(m200_2D))/len(m200_2D)
                            pl.figure()
                            pl.subplot(1,2,1)
                            pl.imshow(mathstools.normalise(m200_2D),interpolation='nearest')
                            pl.subplot(1,2,2)
                            pl.imshow(mathstools.normalise(m200_2D_hires),interpolation='nearest')
                            pl.show()

            else:

                use_prior=False; use_ml=False; use_expectation=False; use_default=False; prior_trick=False
                if args.halo_removal == 'prior': use_prior=True
                if args.halo_removal == 'flat': use_prior=True
                if args.halo_removal == 'ml': use_ml=True
                if args.halo_removal == 'exp': use_expectation=True
                if args.halo_removal == 'default': use_default=True

                if prior_trick:
                    if grid_pickle['grid_h1M200'][0,0,0,0] < 1e12:
                       div=1e14
                    else:
                       div=1
                    
                    # the prior trick
                    select1 = grid_pickle['grid_h1M200'] > (halo1[nf]['m200_fit'] - 3*halo1['m200_errlo'][nf])/div
                    select2 = grid_pickle['grid_h1M200'] < (halo1[nf]['m200_fit'] + 3*halo1['m200_errhi'][nf])/div
                    select3 = grid_pickle['grid_h2M200'] > (halo2[nf]['m200_fit'] - 3*halo2['m200_errlo'][nf])/div 
                    select4 = grid_pickle['grid_h2M200'] < (halo2[nf]['m200_fit'] + 3*halo2['m200_errhi'][nf])/div
                    selectp = select1 & select2 & select3 & select4
                    log_prob_2D = np.zeros([log_prob.shape[0],log_prob.shape[1]])
                    normalisation=log_prob.max()
                    for i1 in range(log_prob.shape[0]):
                        for i2 in range(log_prob.shape[1]):
                            selecti = np.zeros(log_prob.shape,dtype=np.bool)
                            selecti[i1,i2,:,:] = True
                            select = selectp & selecti
                            n_points = len(np.nonzero(select.flatten())[0])
                            if n_points!=0: 
                                all_prob = log_prob[select]  
                                log_prob_2D[i1,i2] = np.log(np.sum(np.exp(all_prob-normalisation))/n_points)
                            else:
                                pdf_prob = np.exp(log_prob - log_prob.max()) 
                                pdf_prob_2D = np.sum(pdf_prob,axis=(2,3))
                                log_prob_2D = np.log(pdf_prob_2D)
                elif use_prior:

                    log_prob = apply_prior(log_prob,grid_pickle['grid_h1M200'],grid_pickle['grid_h2M200'],ids)
                    pdf_prob = np.exp(log_prob - log_prob.max()) 
                    pdf_prob_2D = np.sum(pdf_prob,axis=(2,3))
                    log_prob_2D = np.log(pdf_prob_2D)
                    if np.any(np.isinf(log_prob_2D)) | np.any(np.isnan(log_prob_2D)):
                        import pdb; pdb.set_trace()
                        logger.info('n_nans: %d' % len(np.isnan(log_prob_2D)))
                        logger.info('n_infs: %d' % len(np.isinf(log_prob_2D)))
                        min_element = log_prob_2D[~np.isinf(log_prob_2D)].min()
                        log_prob_2D[np.isinf(log_prob_2D)] = min_element
                        log_prob_2D[np.isnan(log_prob_2D)] = min_element
                elif use_ml:

                    this_h1_m200 = halos['m200_fit'][pairs['ih1'][nf]]
                    this_h2_m200 = halos['m200_fit'][pairs['ih2'][nf]]

                    best_h1m200_index = np.argmin(np.abs(grid_h1M200-this_h1_m200/1e14))  + 1
                    best_h2m200_index = np.argmin(np.abs(grid_h2M200-this_h2_m200/1e14))  + 1
                    if (best_h1m200_index>=len(grid_h1M200)): 
                        best_h1m200_index = len(grid_h1M200)-1
                        logger.warning('ipair=%d really high m200 this_h1_m200=%2.2e',nf,this_h1_m200)
                    if (best_h2m200_index>=len(grid_h2M200)):
                        logger.warning('ipair=%d really high m200 this_h2_m200=%2.2e',nf,this_h2_m200)
                        best_h2m200_index = len(grid_h1M200)-1
                    print '%4d %4d %5.2f %5.2f'  % (best_h1m200_index , best_h2m200_index , grid_h1M200[best_h1m200_index] , grid_h2M200[best_h2m200_index])
                    log_prob_2D = log_prob[:,:,best_h1m200_index,best_h2m200_index]
                    if np.any(np.isinf(log_prob_2D)) | np.any(np.isnan(log_prob_2D)):
                        import pdb; pdb.set_trace()
                        logger.info('n_nans: %d' % len(np.isnan(log_prob_2D)))
                        logger.info('n_infs: %d' % len(np.isinf(log_prob_2D)))
                        min_element = log_prob_2D[~np.isinf(log_prob_2D)].min()
                        log_prob_2D[np.isinf(log_prob_2D)] = min_element
                        log_prob_2D[np.isnan(log_prob_2D)] = min_element

                elif use_expectation:

                    filename_prior = config['filename_halos'].replace('.fits','.prior.pp2')  
                    prior_dict = tabletools.loadPickle(filename_prior,log=0)

                    halo_like1=prior_dict['halos_like'][pairs['ih1'][nf],:]
                    halo_like2=prior_dict['halos_like'][pairs['ih2'][nf],:]
                    halo_like1 = np.exp(halo_like1-halo_like1.max())
                    halo_like2 = np.exp(halo_like2-halo_like2.max())
                    halo_like1 = halo_like1/np.sum(halo_like1)
                    halo_like2 = halo_like2/np.sum(halo_like2)

                    this_h1_m200 = np.sum(halo_like1*prior_dict['grid_M200'])
                    this_h2_m200 = np.sum(halo_like2*prior_dict['grid_M200'])

                    best_h1m200_index = np.argmin(np.abs(grid_h1M200-this_h1_m200/1e14))
                    best_h2m200_index = np.argmin(np.abs(grid_h2M200-this_h2_m200/1e14))
                    print '%4d %4d %5.2f %5.2f'  % (best_h1m200_index , best_h2m200_index , grid_h1M200[best_h1m200_index] , grid_h2M200[best_h2m200_index])

                    log_prob_2D = log_prob[:,:,best_h1m200_index,best_h2m200_index]
                    if np.any(np.isinf(log_prob_2D)) | np.any(np.isnan(log_prob_2D)):
                        import pdb; pdb.set_trace()
                        logger.info('n_nans: %d' % len(np.isnan(log_prob_2D)))
                        logger.info('n_infs: %d' % len(np.isinf(log_prob_2D)))
                        min_element = log_prob_2D[~np.isinf(log_prob_2D)].min()
                        log_prob_2D[np.isinf(log_prob_2D)] = min_element
                        log_prob_2D[np.isnan(log_prob_2D)] = min_element



                elif use_default:

                    total_normalisation += log_prob.max()
                    pdf_prob = np.exp(log_prob - log_prob.max()) 
                    pdf_prob_2D = np.sum(pdf_prob,axis=(2,3))
                    log_prob_2D = np.log(pdf_prob_2D)
                    if np.any(np.isinf(log_prob_2D)) | np.any(np.isnan(log_prob_2D)):
                        import pdb; pdb.set_trace()
                        logger.info('n_nans: %d' % len(np.isnan(log_prob_2D)))
                        logger.info('n_infs: %d' % len(np.isinf(log_prob_2D)))
                        min_element = log_prob_2D[~np.isinf(log_prob_2D)].min()
                        log_prob_2D[np.isinf(log_prob_2D)] = min_element
                        log_prob_2D[np.isnan(log_prob_2D)] = min_element

                else: 
                    raise Exception('halo_removal not test correctly')



            # logprob_kappa0_radius += log_prob_2D + log_prob.max()
            logprob_kappa0_radius += log_prob_2D
            plot_prob_all, _, _, _ = mathstools.get_normalisation(logprob_kappa0_radius)  
            plot_prob_this, _, _, _ = mathstools.get_normalisation(log_prob_2D)   

            if hires:
                # from scipy import interpolate
                # spline = interpolate.bisplrep(grid_kappa0,grid_radius,log_prob_2D,s=0)
                # log_prob_2D_hires = interpolate.bisplev(vec_kappa0_hires,vec_radius_hires,spline)
                func_interp = scipy.interpolate.interp2d(vec_kappa0,vec_radius,log_prob_2D.T, kind='cubic')
                log_prob_2D_hires = func_interp(vec_kappa0_hires,vec_radius_hires).T
                if np.any(np.isnan(log_prob_2D_hires)):
                    print 'nans in log_prob_2D_hires'
                    import pdb; pdb.set_trace()
                # d1 =vec_kappa0[1]-vec_kappa0[0]
                # d2 =vec_radius[1]-vec_radius[0]
                # x=grid_kappa0_hires.flatten()/d1
                # y=(grid_radius_hires.flatten() - min(grid_radius.flatten()))/d2
                # log_prob_2D_hires = np.reshape(bilinear_interpolate(log_prob_2D,x,y),[len(vec_radius_hires),len(vec_radius_hires)]).T
                # log_prob_2D_hires[-1,:] = 2*log_prob_2D_hires[-2,:] - log_prob_2D_hires[-3,:]
                # log_prob_2D_hires[:,-1] = 2*log_prob_2D_hires[:,-2] - log_prob_2D_hires[:,-3]
                logprob_kappa0_radius_hires += log_prob_2D_hires


                prob_1D_hires = np.exp(log_prob_2D_hires-log_prob_2D_hires.max())
                log_prob_1D_hires = np.log(prob_1D_hires.sum(axis=1))
                logprob_kappa0_hires += log_prob_1D_hires

            if plots:
                if nf % 1 == 0:
                    plot_prob_all_hires, _, _,_ = mathstools.get_normalisation(logprob_kappa0_radius_hires)  
                    plot_prob_this_hires, _, _,_ = mathstools.get_normalisation(log_prob_2D_hires)   
                    pl.figure(figsize=(10,10))
                    pl.subplot(2,2,1)
                    pl.pcolormesh(grid_kappa0, grid_radius , plot_prob_all); pl.colorbar()
                    pl.subplot(2,2,2)
                    pl.pcolormesh(grid_kappa0 , grid_radius , plot_prob_this); pl.colorbar()
                    pl.subplot(2,2,3)
                    pl.pcolormesh(grid_kappa0_hires, grid_radius_hires , plot_prob_all_hires); pl.colorbar()
                    pl.subplot(2,2,4)
                    pl.pcolormesh(grid_kappa0_hires , grid_radius_hires , plot_prob_this_hires); pl.colorbar()
                    dtotal = np.sqrt(pairs[nf]['Dxy']**2+pairs[nf]['Dlos']**2)
                    titlestr='nf=%d ih1=%d ih2=%d m200_h1=%2.2e m200_h2=%2.2e sig1=%2.2f sig2=%2.2f Dxy=%2.2f Dlos=%2.2f Dtot=%2.2f' % (nf, pairs[nf]['ih1'] , pairs[nf]['ih2'], pairs[nf]['m200_h1_fit'], pairs[nf]['m200_h2_fit'],halo1['m200_sig'][nf],halo2['m200_sig'][nf],pairs[nf]['Dxy'],pairs[nf]['Dlos'],dtotal)
                    pl.suptitle(titlestr)
                    pl.show()

          
            n_usable_results+=1
            list_ids_used.append(nf)
            logger.debug('%4d %s n_usable_results=%d' , nf , filename_pickle , n_usable_results)
            if nf % 100 == 0 : logger.info('%4d n_usable_results=%d' , nf , n_usable_results)


    if hires:

        grid_radius_hires = grid_radius_hires[:,:]
        grid_kappa0_hires = grid_kappa0_hires[:,:]
        logprob_kappa0_radius_hires = logprob_kappa0_radius_hires[:,:]
        # grid_radius_hires
        grid2D_dict = { 'grid_kappa0'  : grid_kappa0_hires , 'grid_radius' : grid_radius_hires , 'total_normalisation' : total_normalisation}  
        prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob_kappa0_radius_hires)  
        return prod2D_pdf, grid2D_dict, list_ids_used, n_usable_results
    else:
        prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob_kappa0_radius)   
        return prod2D_pdf, grid2D_dict, list_ids_used, n_usable_results 
    
    # return None, None, prod2D_pdf, grid_kappa0, grid_radius, n_usable_results

def get_prob_prod_sampling_2D(ids,plots=False,hires=True,hires_marg=False,normalisation_const=None):
    
    import scipy.interpolate
    n_per_file = 1
    id_file_first = 0
    id_file_last = max(ids)+1
    n_params = 4
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')

    # initialise lists for 1D marginals
    list_pdf_prod_1D = []
    list_pdf_grid_1D = []
   
    n_missing=0
    n_usable_results=0

    vec_kappa0=np.linspace(config['kappa0']['box']['min'],config['kappa0']['box']['max'], config['n_grid'])
    vec_radius=np.linspace(config['radius']['box']['min'],config['radius']['box']['max'], config['n_grid'])
    grid_kappa0,grid_radius=np.meshgrid(vec_kappa0,vec_radius,indexing='ij')

    logprob_kappa0_radius = np.zeros([ len(grid_kappa0[:,0]) , len(grid_radius[0,:]) ])+66
    grid2D_dict = { 'grid_kappa0'  : grid_kappa0 , 'grid_radius' : grid_radius}  
    list_ids_used = []

    normalisation_factor=-10000

    if hires:    
        n_upsample = 10
        vec_kappa0_hires = np.linspace(min(grid_kappa0[:,0]),max(grid_kappa0[:,0]),len(grid_kappa0[:,0])*n_upsample)
        vec_radius_hires = np.linspace(min(grid_radius[0,:]),max(grid_radius[0,:]),len(grid_radius[0,:])*n_upsample)
        grid_kappa0_hires, grid_radius_hires = np.meshgrid(vec_kappa0_hires,vec_radius_hires,indexing='ij')
        logprob_kappa0_radius_hires = np.zeros([ len(grid_kappa0_hires) , len(grid_radius_hires) ])+66

    for nf in range(id_file_first,id_file_last):

        if nf in ids:


            # filename_pickle = 'results_local2scratch/results.prob.%04d.%04d.%s.pp2'  % (nf*n_per_file, (nf+1)*n_per_file , name_data)
            filename_pickle = '%s/results.chain.%04d.%04d.%s.pp2'  % (args.results_dir,nf*n_per_file, (nf+1)*n_per_file , name_data)
            try:
                results_pickle = tabletools.loadPickle(filename_pickle,log=1)
            except:
                logger.info('missing %s' % filename_pickle)
                n_missing +=1
                continue
            if len(results_pickle) == 0:
                logger.info('empty %s' % filename_pickle)
                n_missing +=1
                continue

            # marginal kappa-radius
            # log_prob = results_pickle*214.524/2.577
            marginals = results_pickle['marginals']
            log_prob_2D = marginals[0,1,:,:]

            if np.any(np.isinf(log_prob_2D)) | np.any(np.isnan(log_prob_2D)):
                logger.info('n_nans: %d' % len(np.isnan(log_prob_2D)))
                logger.info('n_infs: %d' % len(np.isinf(log_prob_2D)))
                min_element = log_prob_2D[~np.isinf(log_prob_2D)].min()
                log_prob_2D[np.isinf(log_prob_2D)] = min_element
                log_prob_2D[np.isnan(log_prob_2D)] = min_element

            plot_prob_all, _, _, _ = mathstools.get_normalisation(logprob_kappa0_radius)  
            plot_prob_this, _, _, _ = mathstools.get_normalisation(log_prob_2D)   

            if hires:
                func_interp = scipy.interpolate.interp2d(vec_kappa0,vec_radius,log_prob_2D, kind='cubic')
                log_prob_2D_hires = func_interp(vec_kappa0_hires,vec_radius_hires)
                if np.any(np.isnan(log_prob_2D_hires)):
                    print 'nans in log_prob_2D_hires'
                    import pdb; pdb.set_trace()
                logprob_kappa0_radius_hires += log_prob_2D_hires

            if plots:
                if nf % 10 == 0:
                    plot_prob_all_hires, _, _,_ = mathstools.get_normalisation(logprob_kappa0_radius_hires)  
                    plot_prob_this_hires, _, _,_ = mathstools.get_normalisation(log_prob_2D_hires)   
                    pl.figure(figsize=(10,10))
                    pl.subplot(2,2,1)
                    pl.pcolormesh(grid_kappa0, grid_radius , plot_prob_all); pl.colorbar()
                    pl.subplot(2,2,2)
                    pl.pcolormesh(grid_kappa0 , grid_radius , plot_prob_this); pl.colorbar()
                    pl.subplot(2,2,3)
                    pl.pcolormesh(grid_kappa0_hires, grid_radius_hires , plot_prob_all_hires); pl.colorbar()
                    pl.subplot(2,2,4)
                    pl.pcolormesh(grid_kappa0_hires , grid_radius_hires , plot_prob_this_hires); pl.colorbar()
                    pl.suptitle(nf)
                    pl.show()

          
            n_usable_results+=1
            list_ids_used.append(nf)
            logger.debug('%4d %s n_usable_results=%d' , nf , filename_pickle , n_usable_results)
            if nf % 100 == 0 : logger.info('%4d n_usable_results=%d' , nf , n_usable_results)


    if hires:
        grid2D_dict = { 'grid_kappa0'  : grid_kappa0_hires , 'grid_radius' : grid_radius_hires}  
        prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob_kappa0_radius_hires)  
        return prod2D_pdf, grid2D_dict, list_ids_used, n_usable_results
    else:
        prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob_kappa0_radius)   
        return prod2D_pdf, grid2D_dict, list_ids_used, n_usable_results
    
    # return None, None, prod2D_pdf, grid_kappa0, grid_radius, n_usable_results

def plot_vs_length():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    filename_pickle = args.filename_config.replace('.yaml','.plotdata.length.pp2')
    list_res_dict = tabletools.loadPickle(filename_pickle)
    nx=int(config['n_grid_2D']/2)
    # nx=0
  
    length = pairs['R_pair'] 

    pl.figure()
    pl.hist(length)

    for ib,bin in enumerate(list_res_dict):


        prod2D_pdf = list_res_dict[ib]['prod2D_pdf']
        grid2D_kappa0 = list_res_dict[ib]['grid2D_kappa0']
        grid2D_radius = list_res_dict[ib]['grid2D_radius']

        logger.info('[%2.2e<mass<%2.2e]' % (bin['bin_min'] , bin['bin_max'] ))

        contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod2D_pdf)
        prod2D_pdf,_,_,_ = mathstools.get_normalisation(np.log(prod2D_pdf))

        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)
        pl.colorbar()
        # pl.xlim([0,0.3])
        # pl.ylim([0,2])
        pl.xlabel(r'$\Delta \Sigma 10^{14} M_{*} h \mathrm{Mpc}^{-2}$')
        pl.ylabel('half-mass radius [Mpc/h]')
        title_str= r'%s: filament length $\in [%2.1f , %2.1f]$ Mpc/h , n_pairs=%d' % (args.filename_config.replace('.yaml',''), bin['bin_min'] , bin['bin_max'] , bin['n_pairs_used'] )
        pl.title(title_str)
        pl.xlim([0,0.2])
        pl.ylim([0.25,4])
        filename_fig = 'figs/fig.length.%02d.%s.%d.png' % (ib,args.filename_config.replace('.yaml',''),bin['n_pairs_used'])
        pl.savefig(filename_fig)
        logger.info('saved %s' % filename_fig)


        # pl.figure()
        # X = [grid2D_kappa0, grid2D_radius]
        # y = prod2D_pdf
        # plotstools.plot_dist_meshgrid(X,prod2D_pdf,contour=True,colormesh=True)
        # pl.subplot(2,2,1)
        # # pl.xlim([0,0.3])
        # pl.subplot(2,2,3)
        # # pl.xlim([0,0.3])
        # # pl.ylim([0,2])
        # pl.subplot(2,2,4)
        # pl.xlim([0,2])

        # grid2D_mass = grid2D_kappa0 /  (grid2D_radius * np.pi)
        # pl.figure()
        # pl.scatter(grid2D_mass.flatten(),grid2D_radius.flatten(),40,prod2D_pdf)
        # pl.xlim([0,0.1])
        # pl.ylim([0,2])
        # pl.xlabel("mass")
        # pl.ylabel('half-mass radius [Mpc]')
        # pl.title(str(bin['n_pairs_used']))


    pl.show()    


def plotdata_vs_length():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    bins_edges = [6,9,12,15,18]
    bins_centers = plotstools.get_bins_centers(bins_edges)

    list_res_dict = []

    for ib in range(1,len(bins_edges)):
        length = pairs['R_pair']
        # mass = (halo1['snr']+halo2['snr'])/2.
        ids = np.nonzero((length > bins_edges[ib-1]) * (length < bins_edges[ib]))[0]
        logger.info('bin %d found n=%d ids' % (ib,len(ids)))
        list_prod_pdf , list_grid_pdf , prod2D_pdf ,  grid2D_kappa0 , grid2D_radius , n_pairs_used = get_prob_prod_gridsearch(ids)

        res_dict = {}
        res_dict['ib'] = ib
        res_dict['center'] = bins_centers[ib-1]
        res_dict['n_pairs_used'] = n_pairs_used
        res_dict['list_prod_pdf'] = list_prod_pdf
        res_dict['list_grid_pdf'] = list_grid_pdf
        res_dict['prod2D_pdf'] = prod2D_pdf
        res_dict['grid2D_kappa0'] = grid2D_kappa0
        res_dict['grid2D_radius'] = grid2D_radius
        res_dict['bin_min'] = bins_edges[ib-1]
        res_dict['bin_max'] = bins_edges[ib]

        list_res_dict.append(res_dict)

        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)

    filename_pickle = args.filename_config.replace('.yaml','.plotdata.length.pp2')
    tabletools.savePickle(filename_pickle,list_res_dict)

    pl.show()






def plotdata_vs_mass():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    n_pairs = len(halo1)

    if ('lrgclass' in filename_pairs) or ('clusterz' in filename_pairs):
        bins_snr_edges = [5e13,7.5e13,1e14,2e14,3e14]
        mass_param_name = 'm200'
        mass = (10**halo1[mass_param_name]+10**halo2[mass_param_name])/2.
    elif 'cfhtlens' in filename_pairs:
        bins_snr_edges = [0,6,20]
        mass_param_name = 'snr'
        mass = (halo1[mass_param_name]+halo2[mass_param_name])/2.
    if 'bcc' in filename_pairs: 
        bins_snr_edges = [1e14,2e14,3e14,4e14,5e14]
        mass_param_name = 'm200'
        mass = (halo1[mass_param_name]+halo2[mass_param_name])/2.

    # bins_snr_centers = [ 3 , 6]
    bins_snr_centers = plotstools.get_bins_centers(bins_snr_edges)

    # mass = np.max(np.concatenate([halo1[mass_param_name][:,None],halo2[mass_param_name][:,None]],axis=1),axis=1)
    # pl.hist(mass,bins=np.arange(10)-0.5); pl.show()

    list_res_dict = []

    for ib in range(1,len(bins_snr_edges)):
        # mass = (halo1[mass_param_name]+halo2[mass_param_name])/2.
        # mass = (halo1['snr']+halo2['snr'])/2.

        ids = np.nonzero((mass > bins_snr_edges[ib-1]) * (mass < bins_snr_edges[ib]))[0]

        logger.info('bin %d: [%2.2e<mass<%2.2e], found n=%d ids' % (ib,  bins_snr_edges[ib-1], bins_snr_edges[ib], len(ids)))
        # if ib==1:
        list_prod_pdf , list_grid_pdf , prod2D_pdf ,  grid2D_kappa0 , grid2D_radius , n_pairs_used = get_prob_prod_gridsearch(ids)
        # if ib==2:
            # list_prod_pdf , list_grid_pdf , prod2D_pdf ,  grid2D_kappa0 , grid2D_radius , n_pairs_used = get_prob_prod_gridsearch(ids,plots=True)
        logger.info('using %d pairs' , n_pairs_used)

        res_dict = {}
        res_dict['mass_param_name'] = mass_param_name
        res_dict['ib'] = ib
        res_dict['n_pairs_used'] = n_pairs_used
        res_dict['list_prod_pdf'] = list_prod_pdf
        res_dict['list_grid_pdf'] = list_grid_pdf
        res_dict['prod2D_pdf'] = prod2D_pdf
        res_dict['grid2D_kappa0'] = grid2D_kappa0
        res_dict['grid2D_radius'] = grid2D_radius
        res_dict['bin_min'] = bins_snr_edges[ib-1]
        res_dict['bin_max'] = bins_snr_edges[ib]

        list_res_dict.append(res_dict)



        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)
        pl.title('bin %d: [%2.2e<mass<%2.2e], n=%d ids' % (ib,  bins_snr_edges[ib-1], bins_snr_edges[ib], len(ids)))


    filename_pickle = args.filename_config.replace('.yaml','.plotdata.mass.pp2')
    tabletools.savePickle(filename_pickle,list_res_dict)

    pl.show()

def plot_pickle(filename_pickle):

    # filename_pickle
    # filename_pickle = 'cfhtlens-lrgs.plotdata.mass.pp2'
    # filename_pickle = 'bcc-e.plotdata.mass.pp2'
    pickle=tabletools.loadPickle(filename_pickle)
    prob = pickle[0]['prob']
    grid = pickle[0]['params']
    # X=[grid['grid_kappa0'],grid['grid_radius'],grid['grid_h1M200'],grid['grid_h2M200']]
    X=[grid['grid_kappa0'],grid['grid_radius']]


    plotstools.plot_dist_meshgrid(X,prob,labels=[r"$\Delta\Sigma$  $10^{14} \mathrm{M}_{\odot} \mathrm{Mpc}^{-2} h$",r'radius $\mathrm{Mpc}/h$',r"M200_halo1 $\mathrm{M}_{\odot}/h$",r"M200_halo2 $\mathrm{M}_{\odot}/h$"],contour=True)
    # pl.suptitle('CFHTLens + BOSS-DR10 LRGs, using %d halo pairs' % pickle[0]['n_obj'])
    pl.suptitle('CFHTLens + BOSS-DR10, using %d halo pairs' % pickle[0]['n_obj'])


def plot_vs_mass():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    filename_pickle = args.filename_config.replace('.yaml','.plotdata.mass.pp2')
    list_res_dict = tabletools.loadPickle(filename_pickle)
    nx=int(config['n_grid_2D']/2)

    if 'cfhtlens' in args.filename_config:
        mass_param_name = 'snr'
    if 'bcc' in args.filename_config:
        mass_param_name = 'm200'
    if 'lrgclass' in args.filename_config:
        mass_param_name = 'm200'

       
    for ib,snr_bin in enumerate(list_res_dict):

        logger.info('[%2.2e<mass<%2.2e]' % (snr_bin['bin_min'] , snr_bin['bin_max'] ))

        prod2D_pdf = list_res_dict[ib]['prod2D_pdf']
        grid2D_kappa0 = list_res_dict[ib]['grid2D_kappa0']
        grid2D_radius = list_res_dict[ib]['grid2D_radius']

        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)

        contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod2D_pdf)
        # pl.colorbar()
        cp = pl.contour(grid2D_kappa0 , grid2D_radius, prod2D_pdf,levels=contour_levels,colors='m')
        # pl.clabel(cp, inline=1, fontsize=10

        pl.xlim([0,0.2])
        pl.ylim([0.25,4])
        # pl.xlabel("r'$\Delta \Simga 10^{14} * M_{*} \mathrm{Mpc}^{-2}$'")
        pl.xlabel(r'$\Delta \Sigma 10^{14} M_{*} h \mathrm{Mpc}^{-2}$')
        pl.ylabel('half-mass radius [Mpc/h]')
        title_str= r'%s: mean halo %s $\in [%2.1e , %2.1e]$ , n_pairs=%d' % (args.filename_config.replace('.yaml',''),mass_param_name , snr_bin['bin_min'] , snr_bin['bin_max'] , snr_bin['n_pairs_used'] )
        pl.title(title_str)
        filename_fig = 'figs/fig.mass.%02d.%s.%d.png' % (ib,args.filename_config.replace('.yaml',''),snr_bin['n_pairs_used'])
        pl.savefig(filename_fig)
        logger.info('saved %s' % filename_fig)

        # grid2D_mass = grid2D_kappa0 /  (grid2D_radius * np.pi)
        # pl.figure()
        # pl.scatter(grid2D_mass.flatten(),grid2D_radius.flatten(),40,prod2D_pdf)
        # pl.xlim([0,0.1])
        # pl.ylim([0,2])
        # pl.xlabel("mass")
        # pl.ylabel('half-mass radius [Mpc]')
        # pl.title(str(snr_bin['n_pairs_used']))


    pl.show()

def plot_single_pairs():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    pairs = tabletools.ensureColumn(rec=pairs,name='eyeball_class')

    for ids in range(len(pairs)):

        if pairs['analysis'][ids]!=1:
            continue

        prod_pdf, grid_dict, list_ids_used , n_pairs_used = get_prob_prod_gridsearch_2D([ids])

        Dtot = np.sqrt(pairs[ids]['Dxy']**2+pairs[ids]['Dlos']**2)

        title_str= 'ip=%d ih1=%d ih2=%d m200_h1=%2.2e m200_h2=%2.2e Dxy=%2.2f Dlos=%2.2f Dtot=%2.2f class=%d' % (ids,pairs[ids]['ih1'] , pairs[ids]['ih2'], halo1[ids]['m200_fit'], halo2[ids]['m200_fit'] , pairs[ids]['Dxy'], pairs[ids]['Dlos'], Dtot, pairs['eyeball_class'][ids] )
        logger.info(title_str)
    
        rho_crit = 2.77501358101e+11
        int_DS = 1e14*(grid_dict['grid_kappa0'] * ( grid_dict['grid_radius'] * np.pi ) /2.) 
        d_rs = int_DS / rho_crit
            
        xlabel=r'$\Delta\Sigma$  $10^{14} \mathrm{M}_{\odot} \mathrm{Mpc}^{-2} h$'
        ylabel=r'radius $\mathrm{Mpc}/h$'

        pl.figure(figsize=(10,8))
        pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf)
        # cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='y')
        # pl.colorbar()
        # pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],d_rs,levels=[500,200,100,50,0])

        pl.xlabel(xlabel)
        pl.ylabel(ylabel)
        pl.title(title_str)
        pl.axis('tight')


        filename_fig='figs/pair.%04d.kappa0-radius.png' % ids
        pl.savefig(filename_fig)
        logger.info('saved %s',filename_fig)
        # pl.show()
        pl.close()

def plot_single_pairs_const():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')
    # filename_pairs = config['filename_pairs'].replace('.fits','.addstats.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)
    n_upsample=20

    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')
    filename_grid_const = 'results_const/results.grid.%s.pp2' % (name_data)
    grid_pickle_const = tabletools.loadPickle(filename_grid_const)

    for ids in range(n_pairs):

        filename_pickle_cons = '%s/results.prob.%04d.%04d.%s.pp2'  % ('results_const', ids, ids+1, name_data)
        log_like_const = tabletools.loadPickle(filename_pickle_cons,log=1)


         # constant model
        vec_kappa0_const = grid_pickle_const['grid_kappa0'][:,0,0,0] 
        vec_h1M200_const = grid_pickle_const['grid_h1M200'][0,0,:,0] 
        vec_h2M200_const = grid_pickle_const['grid_h2M200'][0,0,0,:] 
        log_like_const = np.squeeze(log_like_const,1)
        prob_like_const = np.exp(log_like_const - log_like_const.max())
        prob_like_const_1D = np.sum(prob_like_const,axis=(1,2))
        log_like_const_1D = np.log(prob_like_const_1D)
        vec_kappa0_const = np.linspace(vec_kappa0_const.min(),vec_kappa0_const.max(),len(vec_kappa0_const)*n_upsample)
        func_interp = scipy.interpolate.interp1d(np.linspace(0,1,len(prob_like_const_1D)),log_like_const_1D,'cubic')
        log_like_const_1D = func_interp(np.linspace(0,1,len(prob_like_const_1D)*n_upsample))
        # prob_like_const_1D = np.exp(log_like_const_1D-log_prob_2D_hires.max())
        prob_like_const_1D = np.exp(log_like_const_1D)

       
        title_str= 'ih1=%d ih2=%d m200_h1=%2.2f m200_h2=%2.2f class=%d BF=%2.4f' % (pairs[ids]['ih1'] , pairs[ids]['ih2'], pairs[ids]['m200_h1_fit'], pairs[ids]['m200_h2_fit'] , pairs['eyeball_class'][ids] , pairs['bayes_factor'][ids])
        print title_str
            
        xlabel=r'$\Delta\Sigma$  $10^{14} \mathrm{M}_{\odot} \mathrm{Mpc}^{-2} h$'
        ylabel=r'radius $\mathrm{Mpc}/h$'

        pl.figure()
        pl.plot(vec_kappa0_const,prob_like_const_1D)
        pl.xlabel(xlabel)
        pl.title(title_str)
        pl.axis('tight')

        filename_fig='figs/pair.%04d.kappa0-const.png' % ids
        pl.savefig(filename_fig)
        logger.info('saved %s',filename_fig)
        pl.close()


def plotdata_all():

    filename_shears = config['filename_shears']                                  # args.filename_shears 
    filename_pairs = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')
    filename_halos = config['filename_halos']
    # filename_pairs = config['filename_pairs'].replace('.fits','.addstats.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    halos = tabletools.loadTable(filename_halos)
    n_pairs = len(halo1)

    pairs = tabletools.ensureColumn(rec=pairs,name='eyeball_class',dtype='i4')
    if os.path.isfile('class.txt'):
        classification = np.loadtxt('class.txt',dtype='i4')
        pairs['eyeball_class'] = classification[:,1]


    tabletools.saveTable(filename_pairs,pairs)
    filename_cluscat = os.environ['HOME'] + '/data/CFHTLens/Clusters/Clusters.fits'
    cluscat = tabletools.loadTable(filename_cluscat)
    cluscat.dtype.names = [n.lower() for n in cluscat.dtype.names]

    print 'class 0',sum(pairs['eyeball_class']==0)
    print 'class 1',sum(pairs['eyeball_class']==1)
    print 'class 2',sum(pairs['eyeball_class']==2)

    if 'cfhtlens' in filename_pairs:
        bins_snr_edges = [5,20]
        mass_param_name = 'snr'
    else:
        bins_snr_edges = [1e14,1e15]
        mass_param_name = 'm200'
    # bins_snr_centers = [ 3 , 6]
    bins_snr_centers = plotstools.get_bins_centers(bins_snr_edges)

    list_res_dict = []
        
    ids_pairs_use = []
    ids_halos_use = []
    np.sort(pairs,order=['m200_h1_fit','m200_h2_fit'])
    for idc in range(len(pairs)):
        # if (pairs['ih1'][idc] not in ids_halos_use) & (pairs['ih2'][idc] not in ids_halos_use) & (pairs['m200_h1_fit'][idc] > 4e13) & (pairs['m200_h2_fit'][idc] > 4e13):
            ids_pairs_use.append(idc)   
            ids_halos_use.append(pairs['ih1'][idc])
            ids_halos_use.append(pairs['ih2'][idc])
    select = ids_pairs_use
    print select

    for idp in np.array(select).astype(np.int32).copy():
        Dtot = np.sqrt(pairs[idp]['Dxy']**2+pairs[idp]['Dlos']**2)
        # if ( (halo1[idp]['m200_fit'] < 1e14) | (halo2[idp]['m200_fit'] < 6e13) | (Dtot>11) | (pairs[idp]['Dlos']<4) | (pairs[idp]['Dxy']<5) | (halo1[idp]['m200_sig'] < 1) | (halo2[idp]['m200_sig'] < 1)): = 3.01
        # if ( (halo1[idp]['m200_fit'] < 1e14) | (halo2[idp]['m200_fit'] < 1e13) | (Dtot>11) | (pairs[idp]['Dlos']<4) | (pairs[idp]['Dxy']<5) | (halo1[idp]['m200_sig'] < 1) | (halo2[idp]['m200_sig'] < 1) ): = 3.06
        # if ( (halo1[idp]['m200_fit'] < 1e14) | (halo2[idp]['m200_fit'] < 1e13) | (Dtot>11) | (pairs[idp]['Dlos']<3) | (pairs[idp]['Dxy']<5) | (halo1[idp]['m200_sig'] < 0) | (halo2[idp]['m200_sig'] < 0) ): 

            # select.remove(idp)
            # print 'removed ' , idp, len(select)
    
    # remove multiply connected
    # select.remove(17) # 17-18 , 18 is more massive
    # select.remove(68) # 68 - 38 , 38 is more massive
    # select.remove([0,7,10,11,13]) # 73 - 71 , 71 is more massive
    print 'class 0',sum(pairs['eyeball_class'][select]==0)
    print 'class 1',sum(pairs['eyeball_class'][select]==1)
    print 'class 2',sum(pairs['eyeball_class'][select]==2)


    import graphstools
    select_best=graphstools.get_best_neighbour(pairs,halo1,halo2)
    # select_best=graphstools.get_triangulation(pairs,halo1,halo2,halos)

    ids=select_best
    print 'all ids before removing overlaps' , len(ids)
    print ids

    # main
    # 445 216 : 
    # 30 434
    # 341 33
    # 83 82
    # 126 11
    # 60 318 90
    # 68 147
    # 128 118
    # 6 58
    # 259 427
    # -r5: remove 445, 30, 341, 82, 126, 60, 68, 128, 58, 318, 427
    # remove_list = [30 , 445, 341, 82, 126, 90, 60, 68, 128, 58, 427]
    # remove_list = [97 , 118, 364, 26 , 254 , 31 , 345, 394, 237, 123, 259, 357, 317, 359, 11 , 118, 234, 283, 40, 64, 434, 249, 431, 410, 453, 312, 121 ]
    # remove_list = [275 , 236 , 364, 26, 128, 254, 31, 258, 153, 237 , 58, 123, 457, 259, 317, 359, 228, 305, 60, 90, 109, 445, 216, 37, 283, 40, 64, 434, 82, 341, 68, 147, 431, 410, 70, 453, 312, 171 ] # max list
    # remove_list = [81, 97 , 118, 364, 26, 128, 254, 31, 345, 394, 237, 58, 123, 457, 259, 317, 359 , 11, 305, 109, 318, 60, 90, 305, 445, 216, 118, 234, 283, 40, 64, 434, 82, 341, 68, 147, 56, 431, 410, 453, 312, 121  ] # optimal list
    # remove_list = [128 , 118, 216, 6, 318, 60, 90, 30, 82]
    # remove_interloper = [ 4, 31, 11, 216, 44, 21, 445, 341, 147, 431, 427, 105]

    # 40 283
    # 288 11
    # 81 426
    # 145 134
    # 317
    # 97 275
    # 318 60
    # 71 176
    # 81 426
    # 228 11
    # 21 254
    # 236 118
    # 180 37
    # 146 33
    # 111 123

    # remove_list=[317,40,288,426,134,275,60,71,228,21,118,37,146,111] # 011-kappaK rmcloseby
    # remove_list=[317,40,288,426,134,275,60,71,228,21,118,37,146,111] # 011-kappaK rmcloseby
    #185, 285, 
    #186, 286
    #44,43
    #34,33
    #18,301
    #28,27

    # main remove list for 19 pairs
    remove_list=[186] 
    # halo sharing ones : 
    # 43,44 34,33 27,28 18,301
    # remove_list=[186,43,34,27,18]

    # remove list for longer pairs which include short ones
    # 137,35,287 114,457, 216,1   
    # halo sharing ones : 20,19 21,22 35,286 44,43 104,105 14 1 6,7 71 18,301 27,28 34,33
    # remove_list=[186,137,287,114,216,229]     
    # remove_list=[186,137,287,114,216,229,19,21,35,43,105,14,1,7,71,301,27,34]     

    # remove list for longer pairs which > 8mpc
    # remove list: 35,287,459 365,114,457,427 1,216
    # remove_list=[287,43,114,427,216,457]
    # halo sharing ones : 20,19 21,22 35,286 44,43 104,105 14 1 6,7 71 18,301 27,28 34,33
    # remove_list=[186,137,287,114,216,229]     
    # remove_list=[186,137,287,114,216,229,19,21,35,43,105,14,1,7,71,301,27,34]     


    # remove_list=[186,44,34,301,28] 
    # remove_list=[0,318]
    # remove_list=[317,283,288,426,145,275,318,176,228,254,236,180,146,123] # 011-kappaK rmcloseby -- rm smaller
    # remove_list=[317,283,288,426,145,275,318,176,228,254,236,180,146,123] # 011-kappaK rmcloseby 
    # remove_list=[]
    remove_interloper=[]
    for ri in (remove_list+remove_interloper):
        if ri in ids: 
            ids.remove(ri)
            print 'removed ', ri
        else:
            print 'not in pairs' , ri
    # for ri in remove_interloper:
    #     if ri in ids: 
    #         ids.remove([ri])
    #     else:
    #         print 'not in pairs' , ri

    
    # clone2
    # missing=[293-48,301-48,319-48,329-48]
    # ids=np.append(np.arange(288,336),missing)
    # ids=np.arange(288,384) # 013 - good

    # clone1
    # missing=[15+48,16+48,17+48,19+48,22+48,26+48,28+48,29+48,33+48,29+48,34+48,41+48,43+48,45+48]
    # ids= np.nonzero(pairs['ra1']<300)[0]; 
    # ids=ids.tolist()
    # for idh in ids:
    #     if not (((pairs[idh]['ih1']) == (pairs[idh]['ih2']-1)) | ((pairs[idh]['ih2']) == (pairs[idh]['ih1']-1))):
    #         print 'removing' , pairs[idh]['ih1'], pairs[idh]['ih2']
    #         ids.remove(idh)
    # ids= ids[np.arange(0,90)
    # ids=[11]
    # ids.remove(0)
    # ids.remove(10)
    # ids.remove(11)
    # ids.remove(13)

    # random
    # ids= np.random.permutation(1024)

    print len(ids)
    # prod_pdf, grid_dict, n_pairs_used = get_prob_prod_gridsearch(ids)
    if config['optimization_mode'] == 'gridsearch':
        prod_pdf, grid_dict, list_ids_used , n_pairs_used = get_prob_prod_gridsearch_2D(ids)
    if config['optimization_mode'] == 'mcmc':
        prod_pdf, grid_dict, list_ids_used , n_pairs_used = get_prob_prod_sampling_2D(ids)

    select_radius_prior = grid_dict['grid_radius'][0,:]


    for ic in range(len(pairs)):
        Dtot = np.sqrt(pairs['Dxy'][ic]**2+pairs['Dlos'][ic]**2)
        if ic in list_ids_used: 
            mark = '   *' 
        else:
            mark = ''
        # print 'ipair=% 3d\tih1=% 4d\tih2=% 4d\tm200_h1=%2.2e\tm200_h2=%2.2e\tsig1=%2.2f\tsig2=%2.2f\tDxy=%2.2f\tDlos=%2.2f\tDtot=%2.2f\tz=%2.2f\tclass=%d' % (pairs[ic]['ipair'], pairs[ic]['ih1'] , pairs[ic]['ih2'], pairs[ic]['m200_h1_fit'], pairs[ic]['m200_h2_fit'],halo1['m200_sig'][ic],halo2['m200_sig'][ic],pairs['Dxy'][ic],pairs['Dlos'][ic],Dtot,pairs[ic]['z'],pairs[ic]['eyeball_class']) + mark

    print '======================================================'
    print 'used %d pairs' % n_pairs_used

    for ic in list_ids_used:
        Dtot = np.sqrt(pairs[ic]['Dxy']**2+pairs[ic]['Dlos']**2)
        boost = Dtot/pairs[ic]['Dxy']
        print 'ipair=% 3d\tih1=% 4d\tih2=% 4d\th1=[% 5.2f % 5.2f]\th2=[% 5.2f % 5.2f]\tm200_h1=%2.2e\tm200_h2=%2.2e\tsig1=%2.2f\tsig2=%2.2f\tDxy=%2.2f\tDlos=%2.2f\tDtot=%2.2f\tboost=%2.2f\tz=%2.2f\tclass=%d' % (pairs[ic]['ipair'], pairs[ic]['ih1'], pairs[ic]['ih2'], pairs[ic]['ra1'] , pairs[ic]['dec1'], pairs[ic]['ra2'] , pairs[ic]['dec2'], pairs[ic]['m200_h1_fit'], pairs[ic]['m200_h2_fit'],halo1['m200_sig'][ic],halo2['m200_sig'][ic],pairs['Dxy'][ic],pairs['Dlos'][ic],Dtot,boost,pairs[ic]['z'],pairs[ic]['eyeball_class'])
        
    pairs = tabletools.ensureColumn(rec=pairs,name='analysis',dtype='i4')
    pairs['analysis'] = 0
    pairs['analysis'][list_ids_used] = 1
    tabletools.saveTable(filename_pairs,pairs)
    print 'set analysis to 1 for %d pairs' % sum(pairs['analysis']==1)

    print 'class 0' , sum(pairs[list_ids_used]['eyeball_class']==0)
    print 'class 1' , sum(pairs[list_ids_used]['eyeball_class']==1)
    print 'class 2' , sum(pairs[list_ids_used]['eyeball_class']==2)

    res_dict = { 'prob' : prod_pdf , 'params' : grid_dict, 'n_obj' : n_pairs_used }

    list_res_dict.append(res_dict)
    filename_pickle = args.filename_config.replace('.yaml','.plotdata.mass.pp2')
    tabletools.savePickle(filename_pickle,list_res_dict)

    # 2D plot_dist
    plot_pickle(filename_pickle)

    # single 2d plot

    max0, max1 = np.unravel_index(prod_pdf.argmax(), prod_pdf.shape)
    print 'maximum likelihood solution'
    print grid_dict['grid_kappa0'][max0,max1], grid_dict['grid_radius'][max0,max1]
    max_radius = grid_dict['grid_radius'][0,max1]
    max_kappa0 = grid_dict['grid_kappa0'][max0,0]
    contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod_pdf,list_sigmas=[1,2,3,4,5])
    xlabel=r'$\Delta\Sigma$  $10^{14} \mathrm{M}_{\odot} \mathrm{Mpc}^{-2} h$'
    ylabel=r'radius $\mathrm{Mpc}/h$'
    if config['kappa_is_K']:
            xlabel=r' $\Delta\Sigma$ /   $ \mathrm{mean}(\Delta\Sigma_{200})$ '
            ylabel=r' $ r_c / \mathrm{mean}(r_{200}) $'

    
    pl.figure()
    cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='y')
    pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,cmap=pl.get_cmap('Blues'))
    pl.axhline(max_radius,color='r')
    pl.axvline(max_kappa0,color='r')
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    pl.axis('tight')

    # paper plots in ugly colormap
    pl.figure()
    cp = pl.contour(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels,colors='b')
    fmt = {}; strs = [ r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$'] ; 
    for l,s in zip( cp.levels, strs ): 
        fmt[l] = s
    pl.clabel(cp, cp.levels, fmt=fmt)
    # pl.pcolormesh(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,cmap=pl.cm.YlOrRd)
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],1], alpha=0.2 ,  colors=['b'])
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[1],1], alpha=0.2 ,  colors=['b'])
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[2],1], alpha=0.2 ,  colors=['b'])
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[3],1], alpha=0.2 ,  colors=['b'])
    cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[4],1], alpha=0.2 ,  colors=['b'])
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=[contour_levels[0],contour_levels[2]], alpha=0.25 ,  cmap=pl.cm.Blues)
    # cp = pl.contourf(grid_dict['grid_kappa0'],grid_dict['grid_radius'],prod_pdf,levels=contour_levels, alpha=0.25 ,  cmap=pl.cm.bone)
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.title('CFHTLens + BOSS-DR10, using %d halo pairs' % n_pairs_used)
    pl.axis('tight')

    # plot 1d - just kappa0
    prob_kappa0 = np.sum(prod_pdf,axis=1)
    grid_kappa0 = grid_dict['grid_kappa0'][:,0] 
    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_kappa0,prob_kappa0)
    print 'kappa0 using radius marginalised'
    print '%2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

    pl.figure()
    pl.plot(grid_kappa0,prob_kappa0)
    pl.xlabel('kappa0 using radius marginalised')


    prob_radius = np.sum(prod_pdf,axis=0)
    grid_radius = grid_dict['grid_radius'][0,:] 
    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(grid_radius,prob_radius)
    print 'radius using kappa0 marginalised'
    print '%2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

    pl.figure()
    pl.plot(grid_radius,prob_radius)
    pl.xlabel('radius using kappa0 marginalised')

    id_radius=max1
    at_radius=grid_dict['grid_radius'][0,id_radius]
    print 'max kappa0 using radius=' , at_radius
    kappa_at_radius=prod_pdf[:,id_radius].copy()
    kappa_at_radius/=np.sum(kappa_at_radius)
    kappa_grid = grid_dict['grid_kappa0'][:,id_radius]
    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(kappa_grid,kappa_at_radius)
    print '%2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

    pl.figure()
    pl.plot(kappa_grid,kappa_at_radius)
    pl.title('CFHTLens + BOSS-DR10, radius=%2.2f, using %d pairs' % (at_radius,n_pairs_used))
    pl.xlabel(xlabel)

    id_kappa0=max0
    at_kappa0=grid_dict['grid_kappa0'][id_kappa0,0]
    print 'max radius using kappa0=' , at_kappa0
    radius_at_kappa=prod_pdf[id_kappa0,:].copy()
    radius_at_kappa/=np.sum(radius_at_kappa)
    radius_grid = grid_dict['grid_radius'][id_kappa0,:]
    max_par , err_hi , err_lo = mathstools.estimate_confidence_interval(radius_grid,radius_at_kappa)
    print '%2.3f +/- %2.3f %2.3f n_sigma=%2.2f' % (max_par , err_hi , err_lo, max_par/err_lo)

    pl.figure()
    pl.plot(radius_grid,radius_at_kappa)
    # pl.title('CFHTLens + BOSS-DR10, radius=%2.2f, using %d pairs' % (at_kappa0,n_pairs_used))
    pl.xlabel(xlabel)

    print 'mean Dsky'    , np.mean(pairs[ids]['Dxy'])
    print 'mean Dtot'    , np.mean( np.sqrt(pairs[ids]['Dxy']**2+pairs[ids]['Dlos']**2)  )
    print 'mean m200 h1' , np.mean(pairs[ids]['m200_h1_fit'])
    print 'mean m200 h2' , np.mean(pairs[ids]['m200_h2_fit'])
    print 'median z' , np.median(pairs[ids]['z'])

    pl.show()


    # grid_kappa0 = grid_dict['grid_kappa0'][:,:,0,0]
    # grid_radius = grid_dict['grid_radius'][:,:,0,0]
    # prod_pdf_kappa0_radius = np.sum(prod_pdf,axis=(2,3))

    # pl.figure()
    # pl.pcolormesh( grid_kappa0 , grid_radius, prod_pdf_kappa0_radius)

    # contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod_pdf_kappa0_radius)
    # # pl.colorbar()
    # cp = pl.contour(grid_kappa0 , grid_radius, prod_pdf_kappa0_radius,levels=contour_levels,colors='m')
    # # pl.clabel(cp, inline=1, fontsize=10

    # pl.xlim([0,0.3])
    # pl.ylim([0,2])
    # # pl.xlabel("r'$\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}$'")
    # pl.xlabel("'\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}'")
    # pl.ylabel('half-mass radius [Mpc]')
    # pl.axis('tight')
    # title_str= "n_pairs=%d" % (n_pairs_used)
    # pl.title(title_str)
    # filename_fig = 'figs/fig.all.%s.png' % (args.filename_config.replace('.yaml',''))
    # pl.savefig(filename_fig)
    # logger.info('saved %s' % filename_fig)
    # pl.show()

def triangle_plots():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    id_file_first = args.first
    id_file_last = id_file_first + args.num

    filename_grid='%s/results.grid.%s.pp2' % (args.results_dir,os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits',''))
    grid=tabletools.loadPickle(filename_grid)
    X=[grid['grid_kappa0'],grid['grid_radius'],grid['grid_h1M200'],grid['grid_h2M200']]

    if halo2['m200'][0] > 100:
        halo1['m200'] = np.log10(halo1['m200'])
        halo2['m200'] = np.log10(halo2['m200'])

    res_all=None
    n_used=0

    mass= (pairs['m200_h1_fit']+pairs['m200_h2_fit'])/2.
    select = mass > 13.
    # select = pairs['eyeball_class'] == 3

    list_of_ids = range(id_file_first,id_file_last)
    list_of_ids = range(0,66,11)
    for ida in list_of_ids:

        if ~select[ida]:
            continue

        n_used+=1

        print "%d mean_mass=%2.2f halo1_mass=%2.2f halo2_mass=%2.2f z=%2.2f" % (ida,mass[ida],halo1['m200'][ida], halo2['m200'][ida] , halo2[ida]['z'])
        filename_result = '%s/results.prob.%04d.%04d.%s.pp2' % (args.results_dir,ida,ida+1,os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits',''))
        if os.path.isfile(filename_result):
            results_pickle = tabletools.loadPickle(filename_result,log=0)
            res = results_pickle['log_post']
            log_prob_2D = results_pickle['log_post_2D']
            res = apply_prior(res,grid['grid_h1M200'],grid['grid_h2M200'])

        else:
            print 'missing' , filename_result
            continue

        if (res_all !=None):
            if (type(res)!=type(res_all)):
                print 'something wrong with the array', res
                continue
            if (res.shape != res_all.shape):
                print 'something wrong with the shape', res
                continue
       
        res_all = res if res_all == None else res_all+res

    print 'median redshift z=%2.2f' % np.median(halo1['z'][select])
    print 'median mass m=%2.2f' % np.median(mass[select])
    print 'n_used', n_used

    import plotstools, mathstools
    prob=mathstools.normalise(res_all)

    labels=[r"$\Delta \Sigma 10^{14} * M_{\odot} \mathrm{Mpc}^{-2} h$",'radius Mpc/h',r"M200 halo1 $M_{\odot}/h$",r"M200 halo2 $M_{\odot}/h$"]
    mdd = plotstools.multi_dim_dist()
    mdd.labels=labels
    mdd.n_upsample = 10
    mdd.plot_dist_meshgrid(X,prob)
    pl.show()

def plot_data_stamp():


    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    id_pair_first = args.first_result_file
    id_pair_last = args.first_result_file + args.n_results_files 

    for id_pair in range(id_pair_first,id_pair_last):
        

        if '.fits' in config['filename_shears']:
            shears_info = tabletools.loadTable(config['filename_shears'],hdu=id_pair+1)
        elif '.pp2' in config['filename_shears']:
            shears_info = tabletools.loadPickle(config['filename_shears'],pos=id_pair)

        filename_fig = 'figs/fig.stamp.%s.%05d.png' % (os.path.basename(config['filename_shears']).replace('.fits','').replace('.pp2',''),id_pair)
        
        pl.figure(figsize=(10,10))
        pl.subplot(3,1,1)
        filaments_tools.plot_pair(pairs['u1_mpc'][id_pair], pairs['v1_mpc'][id_pair], pairs['u2_mpc'][id_pair], pairs['v2_mpc'][id_pair], shears_info['u_mpc'], shears_info['v_mpc'], shears_info['g1'], shears_info['g2'],idp=id_pair,halo1=halo1[id_pair],halo2=halo2[id_pair],pair_info=pairs[id_pair],plot_type='quiver')
        pl.subplot(3,1,2)
        filaments_tools.plot_pair(pairs['u1_mpc'][id_pair], pairs['v1_mpc'][id_pair], pairs['u2_mpc'][id_pair], pairs['v2_mpc'][id_pair], shears_info['u_mpc'], shears_info['v_mpc'], shears_info['g1'], shears_info['g2'],idp=id_pair,halo1=halo1[id_pair],halo2=halo2[id_pair],pair_info=pairs[id_pair],plot_type='g1')
        pl.subplot(3,1,3)
        filaments_tools.plot_pair(pairs['u1_mpc'][id_pair], pairs['v1_mpc'][id_pair], pairs['u2_mpc'][id_pair], pairs['v2_mpc'][id_pair], shears_info['u_mpc'], shears_info['v_mpc'], shears_info['g1'], shears_info['g2'],idp=id_pair,halo1=halo1[id_pair],halo2=halo2[id_pair],pair_info=pairs[id_pair],plot_type='g2')
        pl.savefig(filename_fig)
        logger.info('saved %s', filename_fig)
        pl.close()


def plot_halo_map():

    filaments_tools.get_halo_map(config['filename_pairs'])

def remove_similar_connections():

    filename_pairs = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    pairs['ipair'] = np.arange(len(pairs))

    n_pairs = len(pairs)
    remove_list = []
    pairs_new = range(len(pairs))
    min_angle = 30 # deg
    min_dist=4
    n_all = 0

    unique_halos, unique_indices = np.unique(np.concatenate([pairs['ih1'],pairs['ih2']]),return_index=True)
    unique_ra = np.concatenate([halo1,halo2])['ra'][unique_indices]
    unique_de = np.concatenate([halo1,halo2])['dec'][unique_indices]
    unique_z = np.concatenate([halo1,halo2])['z'][unique_indices]
    unique_DA =  cosmology.get_ang_diam_dist(unique_z)  
    unique_haloinfo = np.concatenate([halo1,halo2])[unique_indices]
    x,y,z = cosmology.spherical_to_cartesian_with_redshift(unique_ra,unique_de,unique_z)
    n_unique=len(unique_de)
    box_coords = np.concatenate([x[:,None],y[:,None],z[:,None]],axis=1)

    logger.info('getting Ball Tree for 3D')
    BT = BallTree(box_coords, leaf_size=5)
    n_connections=100
    bt_dx,bt_id = BT.query(box_coords,k=n_connections)

    bt_id_reduced = bt_id[:,1:]
    bt_dx_reduced = bt_dx[:,1:]
    ih1 = np.kron( np.ones((n_connections-1,1),dtype=np.int8), np.arange(0,n_unique) ).T
    ih2 = bt_id_reduced
    DA  = bt_dx_reduced

    logger.info('number of pairs %d ' % len(ih1))
    select = ih1 < ih2
    ih1 = ih1[select]
    ih2 = ih2[select]
    DA = DA[select]
    logger.info('number of pairs after removing duplicates %d ' % len(ih1))

    logger.info('calculating x-y distance')
    halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(  unique_ra[ih1] , unique_de[ih1]  )
    halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(  unique_ra[ih2] , unique_de[ih2]  )
    
    min_dist = 1
    # d_xy  = (unique_DA[ih1] + unique_DA[ih2])/2. * cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad)
    d_xy  =  cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad)*180./np.pi

    remove_lrgs = []
    for ip in range(len(d_xy)):
        if ip >= len(d_xy):
            continue
        if d_xy[ip] < min_dist:

            print ih1[ip],ih2[ip],d_xy[ip], unique_haloinfo['dered_r'][ih1[ip]] , unique_haloinfo['dered_r'][ih2[ip]], len(ih1), len(ih2), len(d_xy)
            if unique_haloinfo['dered_r'][ih1[ip]] < unique_haloinfo['dered_r'][ih2[ip]]:
                remove_lrgs.append(unique_halos[ih2[ip]])
                select = (ih1 == ih2[ip]) | (ih2 == ih2[ip])
                ih1  = ih1[~select]
                ih2  = ih2[~select]
                d_xy = d_xy[~select]
            else:
                remove_lrgs.append(unique_halos[ih1[ip]])
                select = (ih1 == ih1[ip]) | (ih2 == ih1[ip])
                ih1  = ih1[~select]
                ih2  = ih2[~select]
                d_xy = d_xy[~select]
            print len(ih1), len(ih2), len(d_xy)
            
    print len(remove_lrgs), len(unique_halos)

    pl.figure()
    pl.scatter(unique_ra,unique_de,c='r')
    pl.scatter(unique_ra[ih1],unique_de[ih1],c='b')
    pl.scatter(unique_ra[ih2],unique_de[ih2],c='c')
    pl.show()

    list_pairs = []
    for ip, vp in enumerate(pairs):
        if (vp['ih1'] in remove_lrgs) or (vp['ih2'] in remove_lrgs):
            remove_list.append(ip)

    for x in set(remove_list): pairs_new.remove(x)
    pairs_prune = pairs[pairs_new]
    halo1_prune = halo1[pairs_new]
    halo2_prune = halo2[pairs_new]

    mass= (pairs_prune['m200_h1_fit'] + pairs_prune['m200_h2_fit'])/2.
    sorting = np.argsort(mass)[::-1]

    pairs_prune = pairs_prune[sorting]
    halo1_prune = halo1_prune[sorting]
    halo2_prune = halo2_prune[sorting]

    filename_pairs_selected = filename_pairs.replace('.fits','.prune.fits')
    filename_halo1_selected = filename_halos1.replace('.halos1.fits','.prune.halos1.fits')
    filename_halo2_selected = filename_halos2.replace('.halos2.fits','.prune.halos2.fits')
    tabletools.saveTable(filename_pairs_selected,pairs_prune)
    tabletools.saveTable(filename_halo1_selected, halo1_prune)
    tabletools.saveTable(filename_halo2_selected, halo2_prune)

    # filaments_tools.get_halo_map(filename_pairs,color='c')
    # filaments_tools.get_halo_map(filename_pairs_selected,color='r')
    # pl.show()

    pairs = pairs_prune
    halo1 = halo1_prune
    halo2 = halo2_prune
    remove_list = []
    pairs_new = range(len(pairs))

    for ic1,vc1 in enumerate(pairs):
        for ic2,vc2 in enumerate(pairs):

            # only one triangle:
            if ic1>ic2:
        
                # connection 1 and 2 have the same node
                same_node=None
                if vc1['ih1'] == vc2['ih1']:
                    same_node_ra , same_node_de = vc2['ra1'] ,  vc2['dec1']
                    other1_ra , other1_de = vc1['ra2'] , vc1['dec2'] 
                    other2_ra , other2_de = vc2['ra2'] , vc2['dec2']
                elif vc1['ih2'] == vc2['ih2']:
                    same_node_ra , same_node_de = vc2['ra2'] ,  vc2['dec2']
                    other1_ra , other1_de = vc1['ra1'] , vc1['dec1'] 
                    other2_ra , other2_de = vc2['ra1'] , vc2['dec1']
                elif vc1['ih1'] == vc2['ih2']:
                    same_node_ra , same_node_de = vc2['ra1'] ,  vc2['dec1']
                    other1_ra , other1_de = vc1['ra2'] , vc1['dec2'] 
                    other2_ra , other2_de = vc2['ra2'] , vc2['dec2']
                else:
                    continue

                # decide if to remove connection or not

                x1=np.array( [other1_ra , other1_de] ) - np.array( [ same_node_ra , same_node_de ] )
                x2=np.array( [other2_ra , other2_de] ) - np.array( [ same_node_ra , same_node_de ] )
                angle = np.arccos( np.dot(x1,x2) / np.linalg.norm(x1) / np.linalg.norm(x2) ) /np.pi * 180

                if angle > min_angle:
                    # print '%4d %4d %4d %4d %5.4f  - saved both %d %d' % (vc1['ih1'],vc1['ih2'],vc2['ih1'],vc2['ih2'],angle,vc1['ipair'],vc2['ipair']) 
                    pass
                else:
                    mass1= vc1['m200_h1_fit'] + vc1['m200_h2_fit']
                    mass2= vc2['m200_h1_fit'] + vc2['m200_h2_fit']
                    if mass1>=mass2:
                        remove_list.append(vc2['ipair'])
                        # print 'removed' , vc1['ipair']
                    else:
                        remove_list.append(vc1['ipair'])
                    n_all+=1
                    if n_all % 100 == 0 : print n_all, n_pairs**2/2., len(remove_list), len(set(remove_list))
                        # print 'removed' , vc2['ipair']

    for ip,vp in enumerate(pairs): 
        if vp['ipair'] in remove_list:
            pairs_new.remove(ip)

    pairs_prune = pairs[pairs_new]
    halo1_prune = halo1[pairs_new]
    halo2_prune = halo2[pairs_new]

    mass= (pairs_prune['m200_h1_fit'] + pairs_prune['m200_h2_fit'])/2.
    sorting = np.argsort(mass)[::-1]

    pairs_prune = pairs_prune[sorting]
    halo1_prune = halo1_prune[sorting]
    halo2_prune = halo2_prune[sorting]

    filename_pairs_selected = filename_pairs.replace('.fits','.prune2.fits')
    filename_halo1_selected = filename_halos1.replace('.halos1.fits','.prune2.halos1.fits')
    filename_halo2_selected = filename_halos2.replace('.halos2.fits','.prune2.halos2.fits')
    tabletools.saveTable(filename_pairs_selected,pairs_prune)
    tabletools.saveTable(filename_halo1_selected, halo1_prune)
    tabletools.saveTable(filename_halo2_selected, halo2_prune)

    filaments_tools.get_halo_map(filename_pairs,color='c')
    filaments_tools.get_halo_map(filename_pairs_selected,color='r')
    pl.show()

def remove_manually():

    filename_pairs = config['filename_pairs']
    pairs = tabletools.loadTable(filename_pairs)

    pairs = tabletools.ensureColumn(rec=pairs,arr=np.zeros(len(pairs)),dtype='i4',name='manual_remove')
    remove_list = [105,250]
    remove_list2 = [46,221,37,82,127]
    pairs['manual_remove'][remove_list]=2
    pairs['manual_remove'][remove_list2]=1


    print 'removed: ', remove_list , remove_list2
    tabletools.saveTable(filename_pairs,pairs)



def snr_analysis():

    filename_pairs =  config['filename_pairs']                                   # pairs_bcc.fits'
    filename_halo1 =  config['filename_pairs'].replace('.fits' , '.halos1.fits') # pairs_bcc.halos1.fits'
    filename_halo2 =  config['filename_pairs'].replace('.fits' , '.halos2.fits') # pairs_bcc.halos2.fits'
    filename_shears = config['filename_shears']                                  # args.filename_shears 

    pairs_table = tabletools.loadTable(filename_pairs)
    halo1_table = tabletools.loadTable(filename_halo1)
    halo2_table = tabletools.loadTable(filename_halo2)

    id_pair = 2
    shears_info = tabletools.loadPickle(filename_shears,id_pair)

    import filaments_model_2hf, filament, nfw
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

    shear_model_g1, shear_model_g2, limit_mask , _ , _  = fitobj.draw_model([0.3, 0.5, 10., 10,])
    pl.figure()
    pl.scatter( fitobj.shear_u_mpc , fitobj.shear_v_mpc , c=shear_model_g1)   
    pl.colorbar()
    pl.figure()
    pl.scatter( fitobj.shear_u_mpc , fitobj.shear_v_mpc , c=shear_model_g2)
    pl.figure()
    fitobj.plot_shears(shear_model_g1,shear_model_g2,limit_mask,quiver_scale=2)

    sigma_g_add =  np.sqrt(1/ np.mean(shears_info['weight']))
    fitobj.shear_g1 =  shear_model_g1 + np.random.randn(len(shears_info['g1']))*sigma_g_add
    fitobj.shear_g2 =  shear_model_g2 + np.random.randn(len(shears_info['g2']))*sigma_g_add
    fitobj.inv_sq_sigma_g = 1./sigma_g_add**2

    n_datapoints = 32
    print np.sqrt(np.sum(shear_model_g1**2)/(sigma_g_add/np.sqrt(n_datapoints))**2)

    import pdb; pdb.set_trace()
    pl.show()
   
def main():


    valid_actions = ['test_kde_methods', 'plot_vs_mass', 'plotdata_vs_mass' , 'plot_vs_length', 'plotdata_vs_length', 'plotdata_all' , 'triangle_plots', 'plot_data_stamp','add_stats' ,'plot_halo_map' , 'plot_pickle','remove_similar_connections' , 'figure_fields_cfhtlens' , 'figure_fields' , 'plot_single_pairs' , 'add_model_selection' , 'mark_overlap' , 'plot_single_pairs_const' , 'remove_manually' , 'snr_analysis' , 'figures_proximity']

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-n', '--num',type=int,  default=-1, action='store', help='number of results files to use')
    parser.add_argument('-f', '--first',type=int, default=0, action='store', help='number of first result file to open')
    parser.add_argument('-p', '--save_plots', action='store_true', help='if to save plots')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )
    parser.add_argument('-rd','--results_dir', action='store', help='where results files are' , default='results/' )
    parser.add_argument('-hr','--halo_removal', action='store', default='prior', choices=('flat','prior','ml','exp' , 'default'), help='which halo removal method to use' )

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


    
if __name__=='__main__':
    main()
