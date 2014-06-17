import sys, os, logging, yaml, argparse, time, tabletools, cosmology, mathstools
import pylab as pl
from scipy.interpolate import interp1d
import numpy as np


logging_level = logging.INFO
log = logging.getLogger("my_script") 
log.setLevel(logging_level)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s", "%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

def get_kl(p,q):

    return np.sum( np.log(p/q) *p )

def run_all():

    bins_z = np.arange(0.025,3.5,0.05)

    filename_gals = '/home/kacprzak/data/CFHTLens/CFHTLens_2014-06-14.normalised.fits'
    filename_clusters = os.environ['HOME'] + '/data/CFHTLens/ClusterZ/clustersz.fits'

    cat_clusters = tabletools.loadTable(filename_clusters)
    cat_gals = tabletools.loadTable(filename_gals)

    gals_ra_deg = cat_gals['ALPHA_J2000']
    gals_de_deg = cat_gals['DELTA_J2000']
    gals_ra_rad , gals_de_rad = cosmology.deg_to_rad(gals_ra_deg, gals_de_deg)

    cylinder_radius_mpc=1

    pz_all=np.sum(cat_gals['PZ_full'],axis=0)
    pz_all=pz_all/np.sum(pz_all)

    n_brigthest = 40
    n_bins_hires = 10000
    bins_z_hires=np.linspace(bins_z.min(), bins_z.max(),n_bins_hires)
    new_z = np.zeros(len(cat_clusters))

 
    for ic in range(len(cat_clusters)):
    # for ic in range(2):

        cluster_ra_rad , cluster_de_rad = cosmology.deg_to_rad( cat_clusters[ic]['ra'] , cat_clusters[ic]['dec'] )
        cluster_z = cat_clusters['z_bad'][ic]
        
        gals_u_rad , gals_v_rad = cosmology.get_gnomonic_projection(gals_ra_rad , gals_de_rad , cluster_ra_rad , cluster_de_rad)
        gals_u_mpc , gals_v_mpc = cosmology.rad_to_mpc(gals_u_rad,gals_v_rad,cluster_z)

        select = (np.sqrt(gals_u_mpc**2 + gals_v_mpc**2) < cylinder_radius_mpc)*( np.abs(cat_gals['Z_B']-cluster_z) < 0.1 )
        # print 'selected %d gals in cylinder' % len(np.nonzero(select)[0])
        cylinder_gals = cat_gals[select]
        gals_u_mpc = gals_u_mpc[select]
        gals_v_mpc = gals_v_mpc[select]

        select1 = (cylinder_gals['MAG_i'] > 10) * (cylinder_gals['MAG_i'] < 27)
        select2 = (cylinder_gals['MAG_r'] > 10) * (cylinder_gals['MAG_r'] < 27)
        select3 = (cylinder_gals['MAG_g'] > 10) * (cylinder_gals['MAG_g'] < 27)
        select4 = (cylinder_gals['MAG_u'] > 10) * (cylinder_gals['MAG_u'] < 27)
        select5 = (cylinder_gals['MAG_y'] > 10) * (cylinder_gals['MAG_y'] < 27)
        select6 = (cylinder_gals['MAG_z'] > 10) * (cylinder_gals['MAG_z'] < 27)

        select = select2*select1*select3*select4*select6
        # print 'selected %d with good mags' % len(np.nonzero(select)[0])
        if len(np.nonzero(select)[0]) == 0:
            print '%d not enough gals' , ic
            import pdb; pdb.set_trace()
            continue
        cylinder_gals = cylinder_gals[select]
        gals_u_mpc = gals_u_mpc[select]
        gals_v_mpc = gals_v_mpc[select]

        x1=cylinder_gals['MAG_r']-cylinder_gals['MAG_i']
        x2=cylinder_gals['MAG_g']-cylinder_gals['MAG_i']
        x3=cylinder_gals['MAG_u']-cylinder_gals['MAG_i']
        x4=cylinder_gals['MAG_y']-cylinder_gals['MAG_i']
        x5=cylinder_gals['MAG_z']-cylinder_gals['MAG_i']
        

        X=np.concatenate( [x1.astype('f4')[:,None], x2.astype('f4')[:,None], x3.astype('f4')[:,None],  x5.astype('f4')[:,None]] ,axis=1)

        from scipy.stats import gaussian_kde
        kde = gaussian_kde(X.T,bw_method=0.3) 
        w = kde(X.T)**3
        w = w/np.max(w)
        # pl.figure()
        # pl.scatter(X[:,0],X[:,1],s=50,c=w) ; pl.colorbar()
        # pl.figure()
        # pl.scatter(X[:,1],X[:,2],s=50,c=w) ; pl.colorbar()
        # pl.figure()
        # pl.scatter(X[:,0],X[:,2],s=50,c=w) ; pl.colorbar()
        # pl.figure()
        # pl.scatter(X[:,0],X[:,3],s=50,c=w) ; pl.colorbar()
        # # pl.figure()
        # # pl.scatter(X[:,0],X[:,4],s=50,c=w) ; pl.colorbar()
        # pl.show()

        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_r']-cylinder_gals[select_brightest]['MAG_i'],  cylinder_gals[select_brightest]['MAG_g']-cylinder_gals[select_brightest]['MAG_i'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()
        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_i'],cylinder_gals[select_brightest]['MAG_r'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()
        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_g'],cylinder_gals[select_brightest]['MAG_r'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()
        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_u'],cylinder_gals[select_brightest]['MAG_i'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()
        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_r'],cylinder_gals[select_brightest]['Z_B'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()

        # pl.figure()
        # pl.scatter(gals_u_mpc,gals_v_mpc,c=cylinder_brightest['Z_B'],s=cylinder_brightest['MAG_r']*2)
        # pl.colorbar()

        # pl.show()

        # pz_hires = np.zeros([len(cylinder_gals),n_bins_hires])
        # for ib in range(len(cylinder_gals)):
        #     fz=interp1d(bins_z,cylinder_gals['PZ_full'][ib],'cubic')
        #     pz_hires[ib,:] = fz(bins_z_hires)
        #     pz_this = pz_hires[ib,:]/np.sum(pz_hires[ib,:])*w[ib]
        #     pl.plot(bins_z_hires,pz_this)
        #     # pl.plot(bins_z,cylinder_gals['PZ_full'][ib],'-');
        #     print 'interp' , ib, np.sum(pz_this)

        # pz_hires[pz_hires<0] = 1e-10
        # pz_prod = np.sum(np.log(pz_hires),axis=0)
        # pz_prod = pz_prod - pz_prod.max()
        # pz_cylinder=np.exp(pz_prod)
        # pz_cylinder=pz_cylinder/np.sum(pz_cylinder)

        # new_z[ic] = bins_z_hires[pz_cylinder.argmax()]
        new_z[ic] = np.sum(cylinder_gals['Z_B']*w)/np.sum(w)
        std_z=np.std(np.sqrt(((cylinder_gals['Z_B']*w - new_z[ic])**2)/np.sum(w)))
        print '%3d new_z=%.4f bad_z=%.4f naomi_z=%.4f n_eff=%2.4f n_cylinder_gals=%d std_z=%2.5f'  % (ic,new_z[ic],cluster_z,cat_clusters['z'][ic],np.sum(w),len(cylinder_gals),std_z)

    tabletools.appendColumn(arr=new_z,rec=cat_clusters,dtype='f4',name='z_est')
    filename_clusters_est = filename_clusters.replace('.fits','.update.fits')
    tabletools.saveTable(filename_clusters_est,cat_clusters)

        # pl.figure()
        # pl.plot(bins_z_hires,pz_cylinder,'kd');
        # # pl.plot(bins_z,pz_all); 
        # pl.axvline(cluster_z,color='b')
        # pl.axvline(cluster_z+0.1,color='r')
        # pl.axvline(cluster_z-0.1,color='r')
        # pl.xlim([0,1])
        # pl.show()

    # pl.hist(new_z-cat_lrgclus['z'],np.linspace(-0.1,0.1,20),histtype='step',label='new z',normed=True);
    # pl.hist(cat_clusters['z']-cat_lrgclus['z'],np.linspace(-0.1,0.1,20),histtype='step',label='naomi z',normed=True);
    # pl.hist(cat_clusters['z_bad']-cat_lrgclus['z'],np.linspace(-0.1,0.1,20),histtype='step',label='old z',normed=True);
    # pl.xlabel('z_estimated - z_spec')
    # pl.legend()
    # pl.show()
    # import pdb; pdb.set_trace()




def run_test():

    bins_z = np.arange(0.025,3.5,0.05)

    filename_gals = '/home/kacprzak/data/CFHTLens/CFHTLens_2014-06-14.normalised.fits'
    filename_clusters = os.environ['HOME'] + '/data/CFHTLens/ClusterZ/clustersz.fits'
    filename_lrgclus = 'halos_cfhtlens_lrgsclus.fits'

    cat_clusters = tabletools.loadTable(filename_clusters)
    cat_lrgclus = tabletools.loadTable(filename_lrgclus)
    cat_gals = tabletools.loadTable(filename_gals)
    cat_clusters = cat_clusters[cat_lrgclus['id']]

    # i=1
    # x=bins_z[20:30]
    # y=cat_gals['PZ_full'][i][20:30]
    # pl.plot(x,y,'o-'); 
    # pl.plot(bins_z,cat_gals['PZ_full'][i],'rx-');

    # xx=np.linspace(x.min(), x.max(),100)
    # f=interp1d(x,y,'cubic')
    # yy=f(xx)
    # pl.plot(xx,yy,'d'); pl.show()

    # import pdb; pdb.set_trace()

    # perm = np.random.permutation(len(cat_gals))[:10000]
    # pl.scatter(cat_gals['ALPHA_J2000'][perm],cat_gals['DELTA_J2000'][perm])

    gals_ra_deg = cat_gals['ALPHA_J2000']
    gals_de_deg = cat_gals['DELTA_J2000']
    gals_ra_rad , gals_de_rad = cosmology.deg_to_rad(gals_ra_deg, gals_de_deg)

    cylinder_radius_mpc=1

    pz_all=np.sum(cat_gals['PZ_full'],axis=0)
    pz_all=pz_all/np.sum(pz_all)

    n_brigthest = 40
    n_bins_hires = 10000
    bins_z_hires=np.linspace(bins_z.min(), bins_z.max(),n_bins_hires)
    new_z = np.zeros(len(cat_lrgclus))

    print 'len(cat_lrgclus)', len(cat_lrgclus)

    for ic in range(len(cat_lrgclus)):

        cluster_ra_rad , cluster_de_rad = cosmology.deg_to_rad( cat_clusters[ic]['ra'] , cat_clusters[ic]['dec'] )
        cluster_z = cat_clusters['z'][ic]
        cluster_zspec = cat_lrgclus['z'][ic]

        gals_u_rad , gals_v_rad = cosmology.get_gnomonic_projection(gals_ra_rad , gals_de_rad , cluster_ra_rad , cluster_de_rad)
        gals_u_mpc , gals_v_mpc = cosmology.rad_to_mpc(gals_u_rad,gals_v_rad,cluster_z)

        select = (np.sqrt(gals_u_mpc**2 + gals_v_mpc**2) < cylinder_radius_mpc)*( np.abs(cat_gals['Z_B']-cluster_z) < 0.1 )
        # print 'selected %d gals in cylinder' % len(np.nonzero(select)[0])
        cylinder_gals = cat_gals[select]
        gals_u_mpc = gals_u_mpc[select]
        gals_v_mpc = gals_v_mpc[select]

        select1 = (cylinder_gals['MAG_i'] > 10) * (cylinder_gals['MAG_i'] < 27)
        select2 = (cylinder_gals['MAG_r'] > 10) * (cylinder_gals['MAG_r'] < 27)
        select3 = (cylinder_gals['MAG_g'] > 10) * (cylinder_gals['MAG_g'] < 27)
        select4 = (cylinder_gals['MAG_u'] > 10) * (cylinder_gals['MAG_u'] < 27)
        select5 = (cylinder_gals['MAG_y'] > 10) * (cylinder_gals['MAG_y'] < 27)
        select6 = (cylinder_gals['MAG_z'] > 10) * (cylinder_gals['MAG_z'] < 27)

        select = select2*select1*select3*select4*select6
        # print 'selected %d with good mags' % len(np.nonzero(select)[0])
        if len(np.nonzero(select)[0]) == 0:
            continue
        cylinder_gals = cylinder_gals[select]
        gals_u_mpc = gals_u_mpc[select]
        gals_v_mpc = gals_v_mpc[select]

        # select_brightest_i = np.ones(len(cylinder_gals))[np.argsort(cylinder_gals['MAG_i'])[:n_brigthest]] == True
        # select_brightest_r = np.ones(len(cylinder_gals))[np.argsort(cylinder_gals['MAG_r'])[:n_brigthest]] == True
        # select_brightest_u = np.ones(len(cylinder_gals))[np.argsort(cylinder_gals['MAG_u'])[:n_brigthest]] == True
        # select_brightest_g = np.ones(len(cylinder_gals))[np.argsort(cylinder_gals['MAG_g'])[:n_brigthest]] == True
        # select_brightest = select_brightest_r 
        # cylinder_brightest = cylinder_gals[select_brightest]
        # gals_u_mpc = gals_u_mpc[select_brightest]
        # gals_v_mpc = gals_v_mpc[select_brightest]
        # print 'using %d gals' % len(cylinder_brightest)

        x1=cylinder_gals['MAG_r']-cylinder_gals['MAG_i']
        x2=cylinder_gals['MAG_g']-cylinder_gals['MAG_i']
        x3=cylinder_gals['MAG_u']-cylinder_gals['MAG_i']
        x4=cylinder_gals['MAG_y']-cylinder_gals['MAG_i']
        x5=cylinder_gals['MAG_z']-cylinder_gals['MAG_i']
        

        X=np.concatenate( [x1.astype('f4')[:,None], x2.astype('f4')[:,None], x3.astype('f4')[:,None],  x5.astype('f4')[:,None]] ,axis=1)

        from scipy.stats import gaussian_kde
        kde = gaussian_kde(X.T,bw_method=0.3) 
        w = kde(X.T)**3
        w = w/np.max(w)
        # pl.figure()
        # pl.scatter(X[:,0],X[:,1],s=50,c=w) ; pl.colorbar()
        # pl.figure()
        # pl.scatter(X[:,1],X[:,2],s=50,c=w) ; pl.colorbar()
        # pl.figure()
        # pl.scatter(X[:,0],X[:,2],s=50,c=w) ; pl.colorbar()
        # pl.figure()
        # pl.scatter(X[:,0],X[:,3],s=50,c=w) ; pl.colorbar()
        # # pl.figure()
        # # pl.scatter(X[:,0],X[:,4],s=50,c=w) ; pl.colorbar()
        # pl.show()

        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_r']-cylinder_gals[select_brightest]['MAG_i'],  cylinder_gals[select_brightest]['MAG_g']-cylinder_gals[select_brightest]['MAG_i'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()
        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_i'],cylinder_gals[select_brightest]['MAG_r'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()
        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_g'],cylinder_gals[select_brightest]['MAG_r'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()
        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_u'],cylinder_gals[select_brightest]['MAG_i'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()
        # pl.figure()
        # pl.scatter(cylinder_gals[select_brightest]['MAG_r'],cylinder_gals[select_brightest]['Z_B'],c=cylinder_gals[select_brightest]['Z_B'])
        # pl.colorbar()

        # pl.figure()
        # pl.scatter(gals_u_mpc,gals_v_mpc,c=cylinder_brightest['Z_B'],s=cylinder_brightest['MAG_r']*2)
        # pl.colorbar()

        # pl.show()

        # pz_hires = np.zeros([len(cylinder_gals),n_bins_hires])
        # for ib in range(len(cylinder_gals)):
        #     fz=interp1d(bins_z,cylinder_gals['PZ_full'][ib],'cubic')
        #     pz_hires[ib,:] = fz(bins_z_hires)
        #     pz_this = pz_hires[ib,:]/np.sum(pz_hires[ib,:])*w[ib]
        #     pl.plot(bins_z_hires,pz_this)
        #     pl.plot(bins_z,cylinder_gals['PZ_full'][ib],'x');
        #     print 'interp' , ib, np.sum(pz_this)

        # pz_hires[pz_hires<0] = 1e-10
        # pz_prod = np.sum(np.log(pz_hires),axis=0)
        # pz_prod = pz_prod - pz_prod.max()
        # pz_cylinder=np.exp(pz_prod)
        # pz_cylinder=pz_cylinder/np.sum(pz_cylinder)

        # new_z[ic] = bins_z_hires[pz_cylinder.argmax()]
        new_z[ic] = np.sum(cylinder_gals['Z_B']*w)/np.sum(w)
        std_z=np.std(np.sqrt(((cylinder_gals['Z_B']*w - new_z[ic])**2)/np.sum(w)))
        print '%3d new_z=%.4f zspec=%.4f bad_z=%.4f naomi_z=%.4f new-zpec=% .4f  naomi-zspec=% .4f n_eff=%2.4f n_cylinder_gals=%d std_z=%2.5f'  % (ic,new_z[ic],cluster_zspec,cluster_z,cat_clusters['z'][ic],new_z[ic]-cluster_zspec,cat_clusters['z'][ic]-cluster_zspec,np.sum(w),len(cylinder_gals),std_z)

        # pl.figure()
        # pl.plot(bins_z_hires,pz_cylinder,'kd');
        # # pl.plot(bins_z,pz_all); 
        # pl.axvline(cluster_z,color='b')
        # pl.axvline(cluster_zspec,color='c')
        # pl.axvline(cluster_z+0.1,color='r')
        # pl.axvline(cluster_z-0.1,color='r')
        # pl.xlim([0,1])
        # pl.show()

    pl.hist(new_z-cat_lrgclus['z'],np.linspace(-0.1,0.1,20),histtype='step',label='new z',normed=True);
    pl.hist(cat_clusters['z']-cat_lrgclus['z'],np.linspace(-0.1,0.1,20),histtype='step',label='naomi z',normed=True);
    pl.hist(cat_clusters['z_bad']-cat_lrgclus['z'],np.linspace(-0.1,0.1,20),histtype='step',label='old z',normed=True);
    pl.xlabel('z_estimated - z_spec')
    pl.legend()
    pl.show()
    import pdb; pdb.set_trace()




        # select all in tube around cluster with dr=5Mpc
            # rotate all galaxies to cluster coordinates with (0,0)
            # calculate x,y,z coords with cluster in 0,0
        # calculate the 




def main():

    global logger , config , args

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
    # config=yaml.load(open(args.filename_config))
    
    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    # run_test()
    run_all()

    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

main()
