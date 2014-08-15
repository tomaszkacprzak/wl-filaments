import pyfits, os, yaml, argparse, sys, logging , cosmology , plotstools , tabletools, matplotlib
import numpy as np
import pylab as pl
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import warnings; warnings.simplefilter('once')

cospars = cosmology.cosmoparams()

# Dxy = R_pair and drloss = Dlos
dtype_pairs = { 'names'   : ['ipair','ih1','ih2','n_gal','DA','Dlos','Dxy','ra_mid','dec_mid','z', 'ra1','dec1','ra2','dec2','u1_mpc','v1_mpc' , 'u2_mpc','v2_mpc' ,'u1_arcmin','v1_arcmin', 'u2_arcmin','v2_arcmin', 'R_pair','drloss','dz','m200_h1_fit','m200_h2_fit','area_arcmin2','n_eff','nu','nv','analysis','eyeball_class'] ,
                'formats' : ['i8']*4 + ['f8']*25 + ['i4']*4 }

dtype_shears_stacked = { 'names' : ['u_mpc','v_mpc','u_arcmin','v_arcmin','g1','g2','mean_scinv', 'g1sc','g2sc','weight',,'weight_sq','n_gals'] , 'formats' : ['f8']*11 + ['i8']*1 }
dtype_shears_single = { 'names' : ['ra_deg','dec_deg','u_mpc','v_mpc','g1','g2', 'g1_orig','g2_orig','scinv','z'] , 'formats' : ['f4']*10 }
dtype_shears_minimal = { 'names' : ['ra_deg','dec_deg','g1_orig','g2_orig','z'] , 'formats' : ['f4']*5 }

# logging.basicConfig(level=logging.INFO,format='%(message)s')
# logger = logging.getLogger("filaments_tools") 

logger = logging.getLogger("fil..tools") 
logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False

redshift_offset = 0.2

config = {}

def get_halo_map(filename_pairs,color=None,pl=pl):

    table_halo1 = tabletools.loadTable(filename_pairs.replace('.fits','.halos1.fits'))
    table_halo2 = tabletools.loadTable(filename_pairs.replace('.fits','.halos2.fits'))
    table_pairs = tabletools.loadTable(filename_pairs)
    print 'before cuts len(table_pairs)=', len(table_pairs)

    import numpy as np
    if 'm200_h1_fit' in table_pairs.dtype.names:
        m200_h1 = table_pairs['m200_h1_fit']
        m200_h2 = table_pairs['m200_h2_fit']
    else:
        m200_h1 = table_pairs['ipair']*0 + 50
        m200_h2 = table_pairs['ipair']*0 + 50
    mean_mass = (m200_h1+m200_h2)/2. 
    select = mean_mass > -100000000000000000.5
    table_halo1 = table_halo1[select]
    table_halo2 = table_halo2[select]
    table_pairs = table_pairs[select]
    m200_h1 = m200_h1[select]
    m200_h2 = m200_h2[select]
    print 'after cuts len(table_pairs)=', len(table_pairs)

    # import numpy as np
    # select = table_halos['M200'] > 1e12
    # table_halo3 = table_halos[select]
    # table_halo3 = table_halo3[np.random.permutation(len(table_halo3))[:10000]]
    # print(len(table_halo3))

    # ra1 , dec1 = cosmology.deg_to_rad( table_pairs['ra1'] , table_pairs['dec1'] )
    # ra2 , dec2 = cosmology.deg_to_rad( table_pairs['ra2'] , table_pairs['dec2'] )
    ra1 , dec1 = table_pairs['ra1'] , table_pairs['dec1'] 
    ra2 , dec2 = table_pairs['ra2'] , table_pairs['dec2'] 

    normalised_z = table_pairs['z'] - min(table_pairs['z'])
    normalised_z/=np.max(normalised_z)

    # from mpl_toolkits.basemap import Basemap
    # import matplotlib.pyplot as plt
    # m = Basemap(projection='ortho',lat_0=-35,lon_0=-10,resolution='c')
    # m.drawparallels(np.arange(-90.,120.,30.))
    # m.drawmeridians(np.arange(0.,420.,60.))
    # m.drawmapboundary()

    # lats = dec1
    # lons = ra1
    # x1,y1 = m(ra1,dec1)
    # x2,y2 = m(ra2,dec2)
    # x3,y3 = m(ra3,dec3)
    # x4,y4 = m(ra4,dec4)

    # def mass(x):
    #     ll  = np.log10(x)
    #     ll[ll==-np.inf] = 0
    #     # return (ll - min(ll) )/ max(ll) * 50. + 10
    #     return ll , (ll - min(ll)) * 100. + 10


    # print max(table_halos['Z']) , min(table_halos['Z'])
    # m.scatter(x1,y1, mass(table_halo1['m200'])[1] , table_halo1['z'] , marker = 'o') #
    # m.scatter(x2,y2, mass(table_halo2['m200'])[1] , table_halo2['z'] , marker = 'o') #
    # m.scatter(x3,y3, mass(table_halo3['M200'])[1] , table_halo3['Z'] , marker = 'o') #

    # m.scatter(x1,y1, 4*m200_h1 , c=table_halo1['z'] , marker = 'o' , cmap=pl.matplotlib.cm.jet) #
    # m.scatter(x2,y2, 4*m200_h2 , c=table_halo2['z'] , marker = 'o' , cmap=pl.matplotlib.cm.jet) #
    # m.scatter(x3,y4, 10 , c=all_halo1['z'] , marker = 'o' ) #
    # m.scatter(x2,y2, 10 , c=all_halo2['z'] , marker = 'o' ) #

    # pl.figure()

    if color==None:
        pl.scatter(ra1,dec1, 4*m200_h1 , c=table_halo1['z'] , marker = 'o' , cmap=matplotlib.cm.jet) #
        pl.scatter(ra2,dec2, 4*m200_h2 , c=table_halo2['z'] , marker = 'o' , cmap=matplotlib.cm.jet) #        
    else:
        pl.scatter(ra1,dec1, 4*m200_h1 , c=color , marker = 'o' , cmap=pl.matplotlib.cm.jet) #
        pl.scatter(ra2,dec2, 4*m200_h2 , c=color , marker = 'o' , cmap=pl.matplotlib.cm.jet) #



    for i in range(len(table_pairs)):
        # m.scatter([x1[i],x2[i]],[y1[i],y2[i]] , c=table_halo2['z'][i] , cmap=pl.matplotlib.cm.jet)
        # m.plot([x1[i],x2[i]],[y1[i],y2[i]],c=normalised_z)
        # pl.plot([ra1[i],ra2[i]],[dec1[i],dec2[i]],c=normalised_z)
        if color==None:
            pl.plot([ra1[i],ra2[i]],[dec1[i],dec2[i]],c='r')
            # pl.text( (ra1[i]+ra2[i])/2. , (dec1[i]+dec2[i])/2., 'r=%2.2fMpc z=%2.2f'%(table_pairs[i]['R_pair'],table_pairs[i]['z']) , fontsize=8 )
        else:
            pl.plot([ra1[i],ra2[i]],[dec1[i],dec2[i]],c=color)
            # pl.text( (ra1[i]+ra2[i])/2. , (dec1[i]+dec2[i])/2., 'r=%2.2fMpc z=%2.2f'%(table_pairs[i]['R_pair'],table_pairs[i]['z']) , fontsize=8 )

    # pl.colorbar()




def get_galaxy_density(shear_ra_deg,shear_de_deg):

    shear_ra_arcmin , shear_de_arcmin = cosmology.deg_to_arcmin(shear_ra_deg , shear_de_deg)

    dra = max(shear_ra_arcmin) - min(shear_ra_arcmin)
    dde = max(shear_de_arcmin) - min(shear_de_arcmin)

    area = dra*dde
    density = float(len(shear_ra_arcmin)) / area
    return density

def wrap_angle_rad(ang_rad):

    if isinstance(ang_rad,np.ndarray):
        select = ang_rad > np.pi
        ang_rad[select] -= 2*np.pi
    elif ang_rad > np.pi:
        ang_rad -= 2*np.pi

    return ang_rad

def wrap_angle_deg(ang_deg):

    if isinstance(ang_deg,np.ndarray):
        select = ang_deg > np.pi
        ang_deg[select] -= 4*90.
    elif ang_deg > np.pi:
        ang_deg -= 4*90.

    return ang_deg


def create_filament_stamp(halo1_ra_deg,halo1_de_deg,halo2_ra_deg,halo2_de_deg,shear_ra_deg,shear_de_deg,shear_g1,shear_g2,shear_z,pair_z,lenscat=None,shear_bias_m=None, shear_weight=None):

        if shear_weight == None:
            shear_weight==np.ones(shear_g1)
        if shear_bias_m == None:
            shear_bias_m==np.zeros(shear_g1)

        # convert to radians
        halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(halo1_ra_deg, halo1_de_deg)
        halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(halo2_ra_deg, halo2_de_deg)
        shear_ra_rad , shear_de_rad = cosmology.deg_to_rad(shear_ra_deg, shear_de_deg)

        # import pylab as pl
        # pl.figure()
        # pl.scatter(halo1_ra_rad     , halo1_de_rad , 100, 'c' , marker='o')
        # pl.scatter(halo2_ra_rad     , halo2_de_rad ,100,  'm' , marker='o')
        # select = np.random.permutation(len(shear_ra_rad))[:10000]
        # pl.scatter(shear_ra_rad[select]  , shear_de_rad[select] ,1,  'm' , marker='.')
        # pl.show()
        # ang_sep = cosmology.get_angular_separation(halo1_ra_rad, halo1_de_rad, halo2_ra_rad, halo2_de_rad)
        # print 'angular sep: ' , ang_sep , ang_sep * cosmology.get_ang_diam_dist(pair_z)

        # get midpoint
        pairs_ra_rad , pairs_de_rad = cosmology.get_midpoint_rad(halo1_ra_rad,halo1_de_rad,halo2_ra_rad,halo2_de_rad)

        # get tangent plane projection        
        shear_u_rad, shear_v_rad = cosmology.get_gnomonic_projection(shear_ra_rad , shear_de_rad , pairs_ra_rad , pairs_de_rad)
        pairs_u_rad, pairs_v_rad = cosmology.get_gnomonic_projection(pairs_ra_rad , pairs_de_rad , pairs_ra_rad , pairs_de_rad)
        halo1_u_rad, halo1_v_rad = cosmology.get_gnomonic_projection(halo1_ra_rad , halo1_de_rad , pairs_ra_rad , pairs_de_rad)
        halo2_u_rad, halo2_v_rad = cosmology.get_gnomonic_projection(halo2_ra_rad , halo2_de_rad , pairs_ra_rad , pairs_de_rad)
        shear_g1_proj , shear_g2_proj = cosmology.get_gnomonic_projection_shear(shear_ra_rad , shear_de_rad , pairs_ra_rad , pairs_de_rad, shear_g1,shear_g2)

        rotation_angle = np.angle(halo1_u_rad + 1j*halo1_v_rad)

        shear_u_rot_rad , shear_v_rot_rad = rotate_vector(rotation_angle, shear_u_rad , shear_v_rad)
        halo1_u_rot_rad , halo1_v_rot_rad = rotate_vector(rotation_angle, halo1_u_rad , halo1_v_rad)
        halo2_u_rot_rad , halo2_v_rot_rad = rotate_vector(rotation_angle, halo2_u_rad , halo2_v_rad)   
        shear_g1_rot , shear_g2_rot = rotate_shear(rotation_angle, shear_u_rad, shear_v_rad, shear_g1_proj, shear_g2_proj)


        # import pylab as pl
        # pl.figure()
        # select = np.random.permutation(len(shear_v_rot_rad))[:10000]
        # pl.scatter( shear_u_rot_rad[select] ,shear_v_rot_rad[select] , 1 , marker=',')
        # pl.scatter( 0     , 0 , 100, 'r' , marker='x')
        # pl.scatter(halo1_u_rot_rad , halo1_v_rot_rad , 100, 'c' , marker='o')
        # pl.scatter(halo2_u_rot_rad , halo2_v_rot_rad , 100, 'm' , marker='o')
        # pl.show()
        # ang_sep = cosmology.get_angular_separation(halo1_u_rot_rad, halo1_v_rot_rad, halo2_u_rot_rad, halo2_v_rot_rad)
        # print 'angular sep: ' , ang_sep , ang_sep * cosmology.get_ang_diam_dist(pair_z)


        # grid boudaries

        # find the angle which corresponds to config['boundary_mpc'] at current redshift
        if config['boundary_mpc'] == '3x4':
            r_pair = np.abs(halo1_u_rot_rad - halo2_u_rot_rad)
            dtheta_x = r_pair * 1.6
            dtheta_y = r_pair * 2.1
            select = (shear_v_rot_rad < dtheta_y) * (shear_v_rot_rad > -dtheta_y) * (shear_u_rot_rad < dtheta_x) *  (shear_u_rot_rad > -dtheta_x)

            range_u_mpc , range_v_mpc = cosmology.rad_to_mpc(2*dtheta_x,2*dtheta_y,pair_z)
            range_u_arcmin , range_v_arcmin = cosmology.rad_to_arcmin(2*dtheta_x,2*dtheta_y)
        
        else:


            dtheta_x = config['boundary_mpc'] / cosmology.get_ang_diam_dist(pair_z) 
            dtheta_y = config['boundary_mpc'] / cosmology.get_ang_diam_dist(pair_z) 

            # select = (shear_v_rot_rad < dtheta_y) * (shear_v_rot_rad > -dtheta_y) * (shear_u_rot_rad < (halo1_u_rot_rad + dtheta_x)) *  (shear_u_rot_rad > (halo2_u_rot_rad - dtheta_x))
            select = ( np.abs( shear_u_rot_rad ) < np.abs(halo1_u_rot_rad + dtheta_x)) * (np.abs(shear_v_rot_rad) < np.abs(dtheta_y)) * (shear_z > (pair_z + redshift_offset))

            range_u_mpc , range_v_mpc = cosmology.rad_to_mpc(2*halo1_u_rot_rad+2*dtheta_x,2*dtheta_y,pair_z)
            range_u_arcmin , range_v_arcmin = cosmology.rad_to_arcmin(2*halo1_u_rot_rad+2*dtheta_x,2*dtheta_y)
        
        logger.info('z=%f dtheta=%f' % (pair_z,dtheta_x))
        logger.info('range_u_mpc=%2.4f range_v_mpc=%2.4f' % (range_u_mpc , range_v_mpc) )
      

        # select the stamp

        shear_u_stamp_rad  = shear_u_rot_rad[select]
        shear_v_stamp_rad  = shear_v_rot_rad[select]
        shear_g1_stamp = shear_g1_rot[select]
        shear_g2_stamp = shear_g2_rot[select]
        shear_g1_orig = shear_g1[select]
        shear_g2_orig = shear_g2[select]
        shear_z_stamp = shear_z[select]
        shear_ra_stamp_deg = shear_ra_deg[select]
        shear_de_stamp_deg = shear_de_deg[select]
        shear_bias_m_stamp = shear_bias_m[select] 
        shear_weight_stamp = shear_weight[select]

        if lenscat != None:
            lenscat_stamp = lenscat[select]
        else:
            lenscat_stamp = None

        if len(shear_z_stamp) == 0:
            import pdb; pdb.set_trace()

        # get Sigma_crit
        sc = cosmology.get_sigma_crit( shear_z_stamp , np.ones(shear_z_stamp.shape)*pair_z , unit=config['Sigma_crit_unit'] )      
        scinv = 1./sc

        g1sc = shear_g1_stamp * sc
        g2sc = shear_g2_stamp * sc
        
        gal_density = get_galaxy_density(shear_ra_stamp_deg,shear_de_stamp_deg)
        logger.info('gal_density %2.2f' % gal_density)

        # logger.info('r_pair=%2.2e rad ' % np.abs(halo1_u_rot_rad - halo2_u_rot_rad))
        # logger.info('dtheta_x=%2.2e rad ' % r_pair)
        # logger.info('getting plot')
        # pl.scatter(shear_u_stamp_rad,shear_v_stamp_rad,c='b')
        # pl.scatter(halo1_u_rot_rad,halo1_v_rot_rad,c='r')
        # pl.scatter(halo2_u_rot_rad,halo2_v_rot_rad,c='r')
        # pl.axis('equal')
        # pl.show()

        # convert to Mpc
        shear_u_stamp_mpc , shear_v_stamp_mpc = cosmology.rad_to_mpc(shear_u_stamp_rad,shear_v_stamp_rad,pair_z)
        halo1_u_rot_mpc , halo1_v_rot_mpc = cosmology.rad_to_mpc(halo1_u_rot_rad,halo1_v_rot_rad,pair_z)
        halo2_u_rot_mpc , halo2_v_rot_mpc = cosmology.rad_to_mpc(halo2_u_rot_rad,halo2_v_rot_rad,pair_z)

        shear_u_stamp_arcmin , shear_v_stamp_arcmin = cosmology.rad_to_arcmin(shear_u_stamp_rad,shear_v_stamp_rad)
        halo1_u_rot_arcmin , halo1_v_rot_arcmin = cosmology.rad_to_arcmin(halo1_u_rot_rad,halo1_v_rot_rad)
        halo2_u_rot_arcmin , halo2_v_rot_arcmin = cosmology.rad_to_arcmin(halo2_u_rot_rad,halo2_v_rot_rad)

        logger.info('r_pair=%2.2fMpc    =%2.2farcmin ' , np.abs(halo1_u_rot_mpc - halo2_u_rot_mpc) , np.abs(halo1_u_rot_arcmin - halo2_u_rot_arcmin))
        logger.info('using %d galaxies for that pair' , len(shear_u_stamp_arcmin))

        
        if config['shear_type'] == 'stacked':
           
            pixel_size_arcmin,_ = cosmology.mpc_to_arcmin(config['pixel_size_mpc'],0.,pair_z) 

            grid_u_mpc = np.arange( -range_u_mpc / 2. , range_u_mpc / 2., config['pixel_size_mpc'] )
            grid_v_mpc = np.arange( -range_v_mpc / 2. , range_v_mpc / 2., config['pixel_size_mpc'] )
            grid_u_arcmin = grid_u_mpc / config['pixel_size_mpc'] * pixel_size_arcmin 
            grid_v_arcmin = grid_v_mpc / config['pixel_size_mpc'] * pixel_size_arcmin


            logger.info('len grid_u_mpc=%d grid_v_mpc=%d' , len(grid_u_mpc) ,  len(grid_v_mpc))
            
            hist_g1, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_g1_stamp * shear_weight_stamp)
            hist_g2, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_g2_stamp * shear_weight_stamp)
            hist_g1sc, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=g1sc * shear_weight_stamp)
            hist_g2sc, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=g2sc * shear_weight_stamp)
            hist_n,  _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) )
            hist_scinv, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=1./sc )
            hist_m , _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=(1+shear_bias_m_stamp) * shear_weight_stamp )
            hist_w , _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_weight_stamp)
            hist_w_sq , _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_weight_stamp**2)

            mean_g1 = hist_g1  / hist_m
            mean_g2 = hist_g2  / hist_m

            mean_scinv = hist_scinv / hist_m
            mean_g1sc = hist_g1sc / hist_m
            mean_g2sc = hist_g2sc / hist_m

            # in case we divided by zero
            mean_g1[hist_n == 0] = 0
            mean_g2[hist_n == 0] = 0
            mean_g1sc[hist_n == 0] = 0
            mean_g2sc[hist_n == 0] = 0
            hist_w[hist_n == 0] = 0
            hist_w_sq[hist_n == 0] = 0
            hist_m[hist_n == 0] = 0

            select = np.isclose(hist_m,0)
            mean_g1[select] = 0
            mean_g2[select] = 0
            mean_g1sc[select] = 0
            mean_g2sc[select] = 0
            hist_w[select] = 0
            hist_w_sq[select] = 0
            hist_m[select] = 0

            u_mid_mpc,v_mid_mpc = plotstools.get_bins_centers(grid_u_mpc) , plotstools.get_bins_centers(grid_v_mpc)
            u_mid_arcmin,v_mid_arcmin = plotstools.get_bins_centers(grid_u_arcmin) , plotstools.get_bins_centers(grid_v_arcmin)
            grid_2d_u_mpc , grid_2d_v_mpc = np.meshgrid(u_mid_mpc,v_mid_mpc,indexing='ij')
            grid_2d_u_arcmin , grid_2d_v_arcmin = np.meshgrid(u_mid_arcmin,v_mid_arcmin,indexing='ij')

            # dtype_shears_stacked = { 'names' : ['u_mpc','v_mpc','u_arcmin','v_arcmin','ra_deg','dec_deg','g1','g2','scinv','weight','z','n_gals'] , 'formats' : ['f8']*11 + ['i8']*1 }
            binned_u_arcmin = grid_2d_u_arcmin.flatten('F')
            binned_v_arcmin = grid_2d_v_arcmin.flatten('F')
            binned_u_mpc = grid_2d_u_mpc.flatten('F')
            binned_v_mpc = grid_2d_v_mpc.flatten('F')
            binned_g1 = mean_g1.flatten('F')
            binned_g2 = mean_g2.flatten('F')
            binned_g1sc = mean_g1sc.flatten('F')
            binned_g2sc = mean_g2sc.flatten('F')
            binned_n = hist_n.flatten('F')
            binned_scinv = hist_scinv.flatten('F')
            binned_w = hist_w.flatten('F')
            binned_w_sq = hist_w.flatten('F')

            u_mpc = binned_u_mpc[:,None]
            v_mpc = binned_v_mpc[:,None]
            u_arcmin = binned_u_arcmin[:,None]
            v_arcmin = binned_v_arcmin[:,None]
            # ra = shear_ra_deg[:,None]
            # de = shear_de_deg[:,None]
            g1 = binned_g1[:,None]
            g2 = binned_g2[:,None]
            g1sc = binned_g1sc[:,None]
            g2sc = binned_g2sc[:,None]

            mean_scinv = binned_scinv[:,None]
            weight = binned_w[:,None]
            weight_sq = binned_w_sq[:,None]
            n_gals = binned_n[:,None] # set all rows to 1
            # scinv = scinv[:,None]
            # z = lenscat_stamp['z'][:,None]
            
            # pl.figure()
            # pl.subplot(1,2,1)
            # pl.imshow(mean_g1,interpolation="nearest")
            # pl.colorbar()
            # pl.subplot(1,2,2)
            # pl.imshow(mean_g2,interpolation="nearest")
            # pl.colorbar()
            # pl.figure()
            # pl.subplot(1,2,1)
            # # plot_pair(halo1_u_rot_mpc , halo1_v_rot_mpc , halo2_u_rot_mpc , halo2_v_rot_mpc , u_mpc, v_mpc, g1sc, g2sc , close=False,nuse = 1,quiver_scale=15)
            # pl.subplot(1,2,2)
            # plot_pair(halo1_u_rot_mpc , halo1_v_rot_mpc , halo2_u_rot_mpc , halo2_v_rot_mpc , shear_u_stamp_mpc, shear_v_stamp_mpc, shear_g1_stamp, shear_g2_stamp , close=False,nuse = 10,quiver_scale=2)
            # pl.show()

            pairs_shear = np.concatenate([u_mpc,v_mpc,u_arcmin,v_arcmin,g1,g2,mean_scinv,g1sc,g2sc,weight,weight_sq,n_gals],axis=1)
            pairs_shear = tabletools.array2recarray(pairs_shear,dtype_shears_stacked)        

            pairs_shear_full = None


        elif config['shear_type'] == 'single':

    
            u_mpc = shear_u_stamp_mpc[:,None]
            v_mpc = shear_v_stamp_mpc[:,None]
            u_arcmin = shear_u_stamp_arcmin[:,None]
            v_arcmin = shear_v_stamp_arcmin[:,None]
            ra = lenscat_stamp['ra'][:,None]
            de = lenscat_stamp['dec'][:,None]
            g1 = shear_g1_stamp[:,None]
            g2 = shear_g2_stamp[:,None]
            g1_orig = shear_g1_orig[:,None]
            g2_orig = shear_g2_orig[:,None]
            scinv = scinv[:,None]
            z = shear_z_stamp[:,None]

            pairs_shear = np.concatenate([ra,de,u_mpc,v_mpc,g1,g2,g1_orig,g2_orig,scinv,z],axis=1)
            pairs_shear = tabletools.array2recarray(pairs_shear,dtype_shears_single)        

            pairs_shear_full = None

        elif config['shear_type'] == 'minimal':

            u_mpc = shear_u_stamp_mpc[:,None]
            v_mpc = shear_v_stamp_mpc[:,None]
            u_arcmin = shear_u_stamp_arcmin[:,None]
            v_arcmin = shear_v_stamp_arcmin[:,None]
            ra = lenscat_stamp['ra'][:,None]
            de = lenscat_stamp['dec'][:,None]
            g1 = shear_g1_stamp[:,None]
            g2 = shear_g2_stamp[:,None]
            g1_orig = shear_g1_orig[:,None]
            g2_orig = shear_g2_orig[:,None]
            scinv = scinv[:,None]
            z = shear_z_stamp[:,None]


            pairs_shear_full = np.concatenate([ra,de,u_mpc,v_mpc,g1,g2,g1_orig,g2_orig,scinv,z],axis=1)
            pairs_shear_full = tabletools.array2recarray(pairs_shear_full,dtype_shears_single)        

            # pairs_shear = np.concatenate([ra,de,g1_orig,g2_orig,scinv,z],axis=1)
            pairs_shear = np.concatenate([ra,de,g1_orig,g2_orig,z],axis=1)
            pairs_shear = tabletools.array2recarray(pairs_shear,dtype_shears_minimal)        

        else: raise ValueError('wrong shear type in config: %s' % config['shear_type'])

        halos_coords = {}
        halos_coords['halo1_u_rot_mpc'] = halo1_u_rot_mpc   
        halos_coords['halo1_v_rot_mpc'] = halo1_v_rot_mpc 
        halos_coords['halo2_u_rot_mpc'] = halo2_u_rot_mpc   
        halos_coords['halo2_v_rot_mpc'] = halo2_v_rot_mpc 
        halos_coords['halo1_u_rot_arcmin'] = halo1_u_rot_arcmin
        halos_coords['halo1_v_rot_arcmin'] = halo1_v_rot_arcmin 
        halos_coords['halo2_u_rot_arcmin'] = halo2_u_rot_arcmin
        halos_coords['halo2_v_rot_arcmin'] = halo2_v_rot_arcmin
        halos_coords['range_u_mpc'] = range_u_mpc
        halos_coords['range_v_mpc'] = range_v_mpc
        halos_coords['range_u_arcmin'] = range_u_arcmin
        halos_coords['range_v_arcmin'] = range_v_arcmin
        halos_coords['gal_density'] = gal_density
        halos_coords['nu'] = len(u_mid_arcmin)
        halos_coords['nv'] = len(v_mid_arcmin)

        return pairs_shear , halos_coords , pairs_shear_full

def rotate_vector(rotation_angle,shear_ra,shear_de):

    shear_pos = (shear_ra + 1j*shear_de)*np.exp(-1j*rotation_angle)
    shear_ra_rot = shear_pos.real 
    shear_de_rot = shear_pos.imag  

    return shear_ra_rot, shear_de_rot

def rotate_shear(rotation_angle,shear_ra,shear_de,shear_g1,shear_g2):

    # shear_pos = (shear_ra + 1j*shear_de)*np.exp(-1j*rotation_angle)
    # shear_ra_rot = shear_pos.real 
    # shear_de_rot = shear_pos.imag
    shear_g = (shear_g1 + 1j*shear_g2)*np.exp(-2j*rotation_angle)
    shear_g1_rot = shear_g.real
    shear_g2_rot = shear_g.imag

    return shear_g1_rot, shear_g2_rot

def plot_pair_quick(pair,shear,idp=0,nuse=10000,filename_fig=None,show=False,close=False,halo1=None,halo2=None,pair_info=None,quiver_scale=2,plot_type='quiver'):

    plot_pair(pair['u1_mpc'],pair['v1_mpc'],pair['u2_mpc'],pair['v2_mpc'],shear['u_mpc'],shear['v_mpc'],shear['g1'],shear['g2'],idp=0,nuse=10000,filename_fig=None,show=False,close=False,halo1=None,halo2=None,pair_info=None,quiver_scale=2,plot_type='quiver')

def plot_pair(halo1_x,halo1_y,halo2_x,halo2_y,shear_x,shear_y,shear_g1,shear_g2,idp=0,nuse=10000,filename_fig=None,show=False,close=False,halo1=None,halo2=None,pair_info=None,quiver_scale=2,plot_type='quiver'):
    """
    In Mpc
    """

    import pylab as pl

    # pl.figure(figsize=(50,25),dpi=500)
    # pl.figure()
    # pl.clf()
    r_pair = 2*np.abs(halo1_x)
    # r_pair = pair_info['R_pair']
    emag=np.sqrt(shear_g1**2+shear_g2**2)
    ephi=0.5*np.arctan2(shear_g2,shear_g1)              

    n_shears_total = len(shear_x)
    if nuse < n_shears_total:
        select = np.random.permutation(n_shears_total)[:nuse]
    else:
        # select all
        select = shear_x < 1e100 
    if plot_type=='quiver':
        pl.quiver(shear_x[select],shear_y[select],emag[select]*np.cos(ephi)[select],emag[select]*np.sin(ephi)[select],linewidths=0.005,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale)
    elif plot_type=='g1':
        pl.scatter(shear_x[select],shear_y[select],50,shear_g1[select],marker='s',edgecolor=None)
    elif plot_type=='g2':
        pl.scatter(shear_x[select],shear_y[select],50,shear_g2[select],marker='s',edgecolor=None)
    elif plot_type=='g':
        pl.scatter(shear_x[select],shear_y[select],50,np.abs(shear_g2[select]+1j*shear_g2[select]),marker=50,edgecolor=None)

    pl.scatter(halo1_x,halo1_y,100,c='b') 
    pl.scatter(halo2_x,halo2_y,100,c='c') 

    if (halo1 != None) and (halo2 != None):
        if 'm200' in halo1.dtype.names: mass1=halo1['m200']
        if 'm200' in halo1.dtype.names: mass2=halo2['m200']
        if 'snr' in halo1.dtype.names: mass1=halo1['snr']
        if 'snr' in halo1.dtype.names: mass2=halo2['snr']
        pl.title('%s r_pair=%2.2fMpc n_gals=%d M1=%2.2e M2=%2.2e' % (idp,r_pair,len(shear_g1),mass1,mass2))
    
    pl.axis('image') 
    # pl.xlim([min(shear_x),max(shear_x)])
    # pl.ylim([min(shear_y),max(shear_y)])

    if filename_fig != None:
        pl.savefig(filename_fig)
        logger.info('saved %s' % filename_fig)
    if show:
        pl.show()
    if close:
        pl.close()




def stats_pairs():

    import pylab as pl
    pairs_table = tabletools.loadTable(config['filename_pairs'])

    logger.info('getting stats')
   
    pl.figure()
    logger.info('distribution of los distance between halos Mpc')
    pl.hist(pairs_table['Dlos'],bins=range(2,20))
    pl.title('distribution of los distance between halos Mpc')
    pl.xlabel('Rlos')
    filename_fig = 'figs/hist.los_pairs.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' % filename_fig)
    pl.close()

    # otherwise it takes forever

    pl.figure()
    logger.info('distribution of length of connections')
    pl.hist(pairs_table['R_pair'],bins=range(2,20))
    pl.title('distribution of length of connections Mpc')
    pl.xlabel('Rpair')
    filename_fig = 'figs/hist.r_pairs.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' % filename_fig)
    pl.close()

    if len(pairs_table) < 1000:
        logger.info('distribution of number of connections per cluster')
        counts = np.array([ sum(pairs_table['ih1']==ih1) for ih1 in pairs_table['ih1']])
        pl.hist(counts,bins=range(0,8))
        pl.title('distribution of number of connections per cluster')
        pl.xlabel('n connections')
        filename_fig = 'figs/hist.n_connections_per_halo.png'
        pl.savefig(filename_fig)
        pl.close()
        logger.info('saved %s' % filename_fig)

# def remove_subhalos(vh1,vh2): 
#     return select

def get_pairs_null1(filename_halos='halos_bcc.fits',filename_pairs_exclude='pairs_bcc.fits',range_Dxy=[6,18],Dlos=6):

    pairs = tabletools.loadTable(filename_pairs_exclude)
    halos = tabletools.loadTable(filename_halos)

    if config['mode'] == 'null1_unpaired':
        select = np.array([ (  (x not in pairs['ih1']) * (x not in pairs['ih2']) ) for x in range(len(halos))])==1
        unpaired_halos = halos[select]
    elif config['mode'] == 'null1_all':
        unpaired_halos = halos.copy()
    
    n_unpaired = len(unpaired_halos)
    logger.info('running in mode %s n_usable_halos %d n_all_halos %d using %d' , config['mode'], len(unpaired_halos) , len(halos) , n_unpaired )

    ipair = np.arange(n_unpaired)
    ih1 = unpaired_halos['id']
    ih2 = np.ones(n_unpaired)*1000
    n_gal = np.zeros(n_unpaired)

    fake_halos = unpaired_halos.copy()


    dtheta = np.random.uniform(low=range_Dxy[0],high=range_Dxy[1],size=n_unpaired) / cosmology.get_ang_diam_dist(fake_halos['z']) 
    dtheta = dtheta * 180. / np.pi
    dalpha = np.random.uniform(low=0,high=np.pi*2)
    dra = (dtheta * np.exp(dalpha*1j) ).real
    ddec = (dtheta * np.exp(dalpha*1j) ).imag

    fake_halos['ra'] = fake_halos['ra'] + dra
    fake_halos['dec'] = fake_halos['dec'] + ddec

    vh1 = unpaired_halos
    vh2 = fake_halos

    halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(vh1['ra'],vh1['dec']) 
    halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(vh2['ra'],vh2['dec'])

    d_xy  = (vh1['DA'] + vh2['DA'])/2. * cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad)
    # true for flat universe
    # d_los = np.abs(vh1['DA'] - vh2['DA'])
    d_los = cosmology.get_ang_diam_dist( vh1['z'] , vh2['z'] )
    DA = d_xy

    ra_mid = (vh1['ra'] + vh2['ra'])/2.
    dec_mid = (vh1['dec'] + vh2['dec'])/2.
    z = (vh1['z'] + vh2['z'])/2.
    R_pair = d_xy
    drloss = d_los
    dz=  np.abs(vh1['z'] - vh2['z'])
    n_gal = dz*0 

    # 

    row = [ipair[:,None],ih1[:,None],ih2[:,None],n_gal[:,None],
            DA[:,None],d_los[:,None],d_xy[:,None],
            ra_mid[:,None],dec_mid[:,None],z[:,None],
            vh1['ra'][:,None],vh1['dec'][:,None],vh2['ra'][:,None],vh2['dec'][:,None],
            ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0, # these fields will be filled in later
            R_pair[:,None],drloss[:,None],dz[:,None]]
    pairs_table = np.concatenate(row,axis=1)
    pairs_table = tabletools.array2recarray(pairs_table,dtype_pairs)
    
    return (pairs_table, vh1, vh2)

def get_pairs_resampling():

    halos = tabletools.loadTable(config['filename_halos'])
    if len(halos) % 2 == 0:
        list_conn1 = range(0,len(halos),2)
        list_conn2 = range(1,len(halos),2)
    else:
        list_conn1 = range(0,len(halos)-1,2)
        list_conn2 = range(1,len(halos),2)

    ih1 = halos[list_conn1]['id']
    ih2 = halos[list_conn2]['id']
    vh1 = halos[list_conn1]
    vh2 = halos[list_conn2]

    halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(  vh1['ra'] ,  vh1['dec']  )
    halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(  vh2['ra'] ,  vh2['dec']  )

    d_xy  = (vh1['DA'] + vh2['DA'])/2. * cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad)
    d_los = cosmology.get_ang_diam_dist( vh1['z'] , vh2['z'] )

    logger.info('neighbour selected min/max d_xy  (%2.2f,%2.2f)'  % (min(d_xy), max(d_xy)) )
    logger.info('neighbour selected min/max d_los (%2.2f,%2.2f)'  % (min(d_los), max(d_los)) )

    ra_mid  = (vh1['ra']  + vh2['ra'] )/2.
    dec_mid = (vh1['dec'] + vh2['dec'])/2.
    z = (vh1['z'] + vh2['z'])/2.
    R_pair = d_xy
    drloss = d_los
    dz=  np.abs(vh1['z'] - vh2['z'])
    n_gal = dz*0 
    DA = cosmology.get_ang_diam_dist(z)
 
    ipair = np.arange(len(ih1))
    pairs_table = np.zeros(len(ipair),dtype=dtype_pairs)
    pairs_table['ipair'] = ipair
    pairs_table['ih1'] = ih1
    pairs_table['ih2'] = ih2
    pairs_table['n_gal'] = 0
    pairs_table['DA'] = DA
    pairs_table['Dlos'] = d_los
    pairs_table['Dxy'] = d_xy
    pairs_table['R_pair'] = d_xy
    pairs_table['ra_mid'] = ra_mid
    pairs_table['dec_mid'] = dec_mid
    pairs_table['z'] = z
    pairs_table['ra1'] = vh1['ra']
    pairs_table['dec1'] = vh1['dec']
    pairs_table['ra2'] = vh2['ra']
    pairs_table['dec2'] = vh2['dec']

    for ip in range(len(pairs_table)):
        pairs_table['m200_h1_fit'][ip] = vh1['m200_fit'][ip]
        pairs_table['m200_h2_fit'][ip] = vh2['m200_fit'][ip]
        print 'halos: % 5d\t% 5d mass= %2.2e %2.2e dx=% 6.2f' % (pairs_table['ih1'][ip],pairs_table['ih2'][ip],pairs_table['m200_h1_fit'][ip],pairs_table['m200_h2_fit'][ip],pairs_table['DA'][ip])

    tabletools.saveTable(config['filename_pairs'],pairs_table)   
    tabletools.saveTable(config['filename_pairs'].replace('.fits','.halos1.fits'), vh1)    
    tabletools.saveTable(config['filename_pairs'].replace('.fits','.halos2.fits'), vh2)    

def get_pairs_topo():

    
    halos_all = tabletools.loadTable(config['filename_halos'])
    halos = halos_all.copy()

    # select on M
    range_M = map(float,config['range_M'])
    select = (halos['m200_fit'] > range_M[0]) * (halos['m200_fit'] < range_M[1])
    halos=halos[select]

    # sort on m200
    halos=halos[halos['m200_fit'].argsort()[::-1]]

    max_angle = config['max_angle']
    max_dx = config['max_dx']

    x,y,z = cosmology.spherical_to_cartesian_with_redshift(halos['ra'],halos['dec'],halos['z'])
    box_coords = np.concatenate( [x[:,None],y[:,None],z[:,None]] , axis=1)
    BT = BallTree(box_coords, leaf_size=5)
    list_conn1 = np.array([],dtype=np.int32)
    list_conn2 = np.array([],dtype=np.int32)
    list_dist  = np.array([],dtype=np.float32)
    for ih,vh in enumerate(halos):
       
        n_connections=20
        this_coords = box_coords[ih,:][:,None].T
        bt_dx,bt_id = BT.query(this_coords,k=n_connections)
        i1 = ih
        all_id = bt_id[0,1:]
        all_dx = bt_dx[0,1:]
        for ic,vc in enumerate(all_id):
            
            i2 = vc
            dx = all_dx[ic]

            add_so_far = True
            if (len(list_conn1)==0) & (dx < max_dx):
                list_conn1=np.append(list_conn1,i1)
                list_conn2=np.append(list_conn2,i2)
                list_dist=np.append(list_dist,dx)
                print '% 4d adding % 4d d=%2.2f' % (i1 , i2 , dx)
            else:
                
                # pull all connections to this halo
                select = np.nonzero( (( list_conn1 == i1 ) | (list_conn2 == i1)) )[0] 

                for j in select:
                    i3 = list_conn1[j] if list_conn1[j]!=i1 else list_conn2[j]
                    if i3 == i2:
                        print '% 4d --- % 4d vs % 4d angle %2.2f -- already exists' % (i1, i2, i3 , angle)
                        add_so_far=False
                        continue

                    x2 = box_coords[i2,:] - box_coords[i1,:]
                    x3 = box_coords[i3,:] - box_coords[i1,:]
                    angle = np.arccos(np.dot(x2, x3) / (np.linalg.norm(x2) * np.linalg.norm(x3))) * 180. / np.pi
                    print '% 4d --- % 4d vs % 4d angle %2.2f' % (i1, i2, i3 , angle)
                    if (angle < max_angle) | (dx > max_dx): 
                        add_so_far=False
                        print '% 4d skipping % 4d d=%2.2f' % (i1 , i2 , dx)


                if add_so_far & (dx < max_dx):
                    list_conn1=np.append(list_conn1,i1)
                    list_conn2=np.append(list_conn2,i2)                     
                    list_dist=np.append(list_dist,dx)
                    print '% 4d adding % 4d d=%2.2f' % (i1 , i2 , dx)

        print 'n_conn %d' % len(list_conn1)


    ih1 = halos[list_conn1]['id']
    ih2 = halos[list_conn2]['id']
    vh1 = halos_all[ih1]
    vh2 = halos_all[ih2]

    halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(  vh1['ra'] ,  vh1['dec']  )
    halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(  vh2['ra'] ,  vh2['dec']  )

    d_xy  = (vh1['DA'] + vh2['DA'])/2. * cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad)
    d_los = cosmology.get_ang_diam_dist( vh1['z'] , vh2['z'] )

    logger.info('neighbour selected min/max d_xy  (%2.2f,%2.2f)'  % (min(d_xy), max(d_xy)) )
    logger.info('neighbour selected min/max d_los (%2.2f,%2.2f)'  % (min(d_los), max(d_los)) )

    ra_mid  = (vh1['ra']  + vh2['ra'] )/2.
    dec_mid = (vh1['dec'] + vh2['dec'])/2.
    z = (vh1['z'] + vh2['z'])/2.
    R_pair = d_xy
    drloss = d_los
    dz=  np.abs(vh1['z'] - vh2['z'])
    n_gal = dz*0 
 
    ipair = np.arange(len(ih1))
    pairs_table = np.zeros(len(ipair),dtype=dtype_pairs)
    pairs_table['ipair'] = ipair
    pairs_table['ih1'] = ih1
    pairs_table['ih2'] = ih2
    pairs_table['n_gal'] = 0
    pairs_table['DA'] = list_dist
    pairs_table['Dlos'] = d_los
    pairs_table['Dxy'] = d_xy
    pairs_table['R_pair'] = d_xy
    pairs_table['ra_mid'] = ra_mid
    pairs_table['dec_mid'] = dec_mid
    pairs_table['z'] = z
    pairs_table['ra1'] = vh1['ra']
    pairs_table['dec1'] = vh1['dec']
    pairs_table['ra2'] = vh2['ra']
    pairs_table['dec2'] = vh2['dec']

    for ip in range(len(pairs_table)):
        pairs_table['m200_h1_fit'][ip] = vh1['m200_fit'][ip]
        pairs_table['m200_h2_fit'][ip] = vh2['m200_fit'][ip]
        print 'halos: % 5d\t% 5d mass= %2.2e %2.2e dx=% 6.2f' % (pairs_table['ih1'][ip],pairs_table['ih2'][ip],pairs_table['m200_h1_fit'][ip],pairs_table['m200_h2_fit'][ip],pairs_table['DA'][ip])

    tabletools.saveTable(config['filename_pairs'],pairs_table)   
    tabletools.saveTable(config['filename_pairs'].replace('.fits','.halos1.fits'), vh1)    
    tabletools.saveTable(config['filename_pairs'].replace('.fits','.halos2.fits'), vh2)    

def get_pairs_new():

    range_Dxy = map(float,config['range_Dxy'])
    range_M = map(float,config['range_M'])
    Dlos = config['max_dlos']
    range_z = map(float,config['range_z'])
  
    halocat = tabletools.loadTable(config['filename_halos'])
    logger.info('number of halos: %d' % len(halocat))  

    # select on M
    select = (halocat['m200_fit'] > range_M[0]) * (halocat['m200_fit'] < range_M[1])
    halocat=halocat[select]
    halocat = tabletools.appendColumn(rec=halocat,arr=np.ones(len(halocat)),dtype='i4',name='is_lrg')
    halocat = tabletools.appendColumn(rec=halocat,arr=np.arange(len(halocat)),dtype='i4',name='halo_id')
    halocat = tabletools.appendColumn(rec=halocat,arr=np.ones(len(halocat))*-1,dtype='i4',name='clus_id')
    logger.info('selected on SNR number of halos: %d' % len(halocat))

    cluscat = tabletools.loadTable(config['filename_cfhtlens_clusters'])   
    cluscat_append = np.zeros(len(cluscat),dtype=halocat.dtype)
    cluscat_append['ra'] = cluscat['RA']
    cluscat_append['dec'] = cluscat['DEC']
    cluscat_append['z'] = cluscat['z']
    cluscat_append['m200_fit'] = 10**cluscat['m200']
    cluscat_append['is_lrg'] = 0
    cluscat_append['clus_id'] = cluscat['id']
    cluscat_append['halo_id'] = -1

    # join
    halocat = np.concatenate([halocat,cluscat_append])

    # select by mass
    select = (halocat['m200_fit'] > range_M[0]) * (halocat['m200_fit'] < range_M[1])
    halocat=halocat[select]      
    halocat['index'] =range(len(halocat))

    # for each LRG add three nearest clusters
    pairs_list = []
    indices_list = []
    n_connections = 50
    redshift_error = 0.15
    n_closest = 3
    min_ang_sep_mpc =1
    ipair = -1
    for ic in range(len(halocat)):
        this_halo = halocat[ic]
        ang_sep = cosmology.get_angular_separation(this_halo['ra'],this_halo['dec'],halocat['ra'],halocat['dec'],unit='deg')
        sorting = ang_sep.argsort()[1:]
        halocat_sorted = halocat[sorting]
        select = np.abs(this_halo['z'] - halocat_sorted['z']) < redshift_error
        n_add=0
        for nc in range(len(halocat_sorted[select])):
            if n_add >= n_closest:
                break
            this_close = halocat_sorted[nc]
            ang_sep = cosmology.get_angular_separation(this_halo['ra'],this_halo['dec'],this_close['ra'],this_close['dec'],unit='deg') 
            z_sep = np.abs(this_halo['z']-this_close['z'])
            ang_sep_mpc = ang_sep*np.pi/180 * cosmology.get_ang_diam_dist(this_halo['z'])
            if ang_sep_mpc>min_ang_sep_mpc:
                n_add += 1
                logger.info('% 6d - % 6d \tang_sep=%6.2f ang_sep_mpc=%2.2f z_esp=%2.2f m1=%2.2e m2=%2.2e %d %d' % (this_halo['index'],this_close['index'],ang_sep,ang_sep_mpc,z_sep,this_halo['m200_fit'],this_close['m200_fit'],this_halo['is_lrg'],this_close['is_lrg']))
                # add more Dlos and Dxy conditions
                ipair+=1
                row = np.zeros(1,dtype=dtype_pairs)
                row['ipair'] = ipair
                row['ih1'] = this_halo['index']
                row['ih2'] = this_close['index']
                row['ra1'] = this_halo['ra']
                row['dec1'] = this_halo['dec']
                row['ra2'] = this_close['ra']
                row['dec2'] = this_close['dec']
                row['z'] = this_close['z']
                logger.info('ra1=%2.3f dec1=%2.3f ra2=%2.3f dec2=%2.3f',row['ra1'],row['dec1'],row['ra2'],row['dec2'])

                indices = [int(row['ih1']),int(row['ih2'])]
                indices.sort()
                if ipair==0:
                    indices_list.append(indices)
                else:
                    if indices not in indices_list:
                        indices_list.append(indices)
                        pairs_list.append(row)
                    else:
                        print 'skipping' , indices

    pairs_table = np.concatenate(pairs_list)
    vh1 = halocat[pairs_table['ih1']]
    vh2 = halocat[pairs_table['ih2']]

    pairs_table = tabletools.ensureColumn(rec=pairs_table,name='eyeball_class',dtype='i4')
    tabletools.saveTable(config['filename_pairs'],pairs_table)   
    tabletools.saveTable(config['filename_pairs'].replace('.fits','.halos1.fits'), vh1)    
    tabletools.saveTable(config['filename_pairs'].replace('.fits','.halos2.fits'), vh2)    

    import pdb; pdb.set_trace()





def get_pairs():

    range_Dxy = map(float,config['range_Dxy'])
    range_M = map(float,config['range_M'])
    Dlos = config['max_dlos']
    range_z = map(float,config['range_z'])
  
    halocat = tabletools.loadTable(config['filename_halos'])
    logger.info('number of halos: %d' % len(halocat))  

    # select on M
    select = (halocat['m200_fit'] > range_M[0]) * (halocat['m200_fit'] < range_M[1])
    halocat=halocat[select]
    logger.info('selected on SNR number of halos: %d' % len(halocat))

    import graphstools
    X = np.concatenate( [ np.arange(len(halocat))[:,None] , halocat['ra'][:,None] , halocat['dec'][:,None] , halocat['z'][:,None], halocat['m200_fit'][:,None], np.zeros(len(halocat))[:,None] ],axis=1 )
    select = graphstools.get_graph(X,min_dist=config['graph_min_dist_deg'],min_z=config['graph_min_dist_z'])
    halocat = halocat[select]
    logger.info('number of halos after graph selection %d', len(halocat))

    logger.info('getting euclidian coords')
    x=halocat['xphys'][:,None]
    y=halocat['yphys'][:,None]
    z=halocat['zphys'][:,None]
    box_coords = np.concatenate( [x,y,z] , axis=1)
  
    logger.info('getting Ball Tree for 3D')
    BT = BallTree(box_coords, leaf_size=5)
    n_connections=70
    bt_dx,bt_id = BT.query(box_coords,k=n_connections)

    increase_range = 1.5
    min_separation = range_Dxy[0]/increase_range
    max_separation = range_Dxy[1]*increase_range
    bt_id_reduced = bt_id[:,1:]
    bt_dx_reduced = bt_dx[:,1:]
    select1=bt_dx_reduced < max_separation 
    select2=bt_dx_reduced > min_separation 
    select = select1 * select2
    conn1 = np.kron( np.ones((n_connections-1,1),dtype=np.int8), np.arange(0,len(halocat)) ).T
    ih1 = conn1[select]
    ih2 = bt_id_reduced[select]
    DA  = bt_dx_reduced[select]

    logger.info('number of pairs %d ' % len(ih1))
    select = ih1 < ih2
    ih1 = ih1[select]
    ih2 = ih2[select]
    DA = DA[select]
    logger.info('number of pairs after removing duplicates %d ' % len(ih1))

    logger.info('calculating x-y distance')
    vh1 = halocat[ih1]
    vh2 = halocat[ih2]
    halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(  vh1['ra'] ,  vh1['dec']  )
    halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(  vh2['ra'] ,  vh2['dec']  )

    d_xy  = (vh1['DA'] + vh2['DA'])/2. * cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad)
    # true for flat universe
    # d_los = np.abs(vh1['DA'] - vh2['DA'])
    d_los = cosmology.get_ang_diam_dist( vh1['z'] , vh2['z'] )

    logger.info('neighbour selected min/max d_xy  (%2.2f,%2.2f)'  % (min(d_xy), max(d_xy)) )
    logger.info('neighbour selected min/max d_los (%2.2f,%2.2f)'  % (min(d_los), max(d_los)) )

    max_los = Dlos
    # select the plane separation
    logger.debug('select the plane separation')
    select1 = np.abs(d_los) < max_los
    select2 = d_xy > range_Dxy[0]
    select3 = d_xy < range_Dxy[1]



    # select
    select  = select1 * select2 * select3
    ih1 = ih1[select]
    ih2 = ih2[select]
    DA = DA[select]
    vh1 = halocat[ih1]
    vh2 = halocat[ih2]
    d_los=d_los[select]
    d_xy=d_xy[select]
    logger.info('number of pairs %d ' % len(ih1))
    logger.info('final min/max d_xy  (%2.2f,%2.2f)'  % (min(d_xy), max(d_xy)) )
    logger.info('final min/max d_los (%2.2f,%2.2f)'  % (min(d_los), max(d_los)) )

    ra_mid = (vh1['ra'] + vh2['ra'])/2.
    dec_mid = (vh1['dec'] + vh2['dec'])/2.
    z = (vh1['z'] + vh2['z'])/2.
    R_pair = d_xy
    drloss = d_los
    dz=  np.abs(vh1['z'] - vh2['z'])
    n_gal = dz*0 

    ipair = np.arange(len(ih1))
    # dtype_pairs = { 'names'   : ['ipair','ih1','ih2','n_gal','DA','Dlos','Dxy','ra_mid','dec_mid','z', 'ra1','dec1','ra2','dec2','u1_mpc','v1_mpc' , 'u2_mpc','v2_mpc' ,'u1_arcmin','v1_arcmin', 'u2_arcmin','v2_arcmin', 'R_pair','drloss','dz','area_arcmin2','n_eff','nu','nv'] ,

    pairs_table = np.zeros(len(ipair),dtype=dtype_pairs)
    pairs_table['ipair'] = ipair
    pairs_table['ih1'] = ih1
    pairs_table['ih2'] = ih2
    pairs_table['n_gal'] = n_gal
    pairs_table['DA'] = DA
    pairs_table['Dlos'] = d_los
    pairs_table['Dxy'] = d_xy
    pairs_table['R_pair'] = d_xy
    pairs_table['ra_mid'] = ra_mid
    pairs_table['dec_mid'] = dec_mid
    pairs_table['z'] = z
    pairs_table['ra1'] = vh1['ra']
    pairs_table['dec1'] = vh1['dec']
    pairs_table['ra2'] = vh2['ra']
    pairs_table['dec2'] = vh2['dec']

    import graphstools
    select = graphstools.remove_similar_connections(pairs_table,min_angle=config['graph_min_angle'])

    logger.info('number of pairs before removing similar connections = %d' , len(pairs_table))
    pairs_table = pairs_table[select]
    vh1 = vh1[select]
    vh2 = vh2[select]
    logger.info('number of pairs before after similar connections = %d' , len(pairs_table))
    pairs_table['ipair'] = range(len(pairs_table))

    pairs_table=tabletools.ensureColumn(rec=pairs_table,name='m200_h1_fit')
    pairs_table=tabletools.ensureColumn(rec=pairs_table,name='m200_h2_fit')
    if 'm200_fit' in vh1.dtype.names:
        m200_col = 'm200_fit' 
    elif 'm200' in vh1.dtype.names:
        m200_col = 'm200' 
    for ip in range(len(pairs_table)):
        pairs_table['m200_h1_fit'][ip] = vh1[m200_col][ip]
        pairs_table['m200_h2_fit'][ip] = vh2[m200_col][ip]
        print 'halos: % 4d % 4d mass= %2.2e %2.2e' % (pairs_table['ih1'][ip],pairs_table['ih2'][ip],pairs_table['m200_h1_fit'][ip],pairs_table['m200_h2_fit'][ip])

    tabletools.saveTable(config['filename_pairs'],pairs_table)   
    tabletools.saveTable(config['filename_pairs'].replace('.fits','.halos1.fits'), vh1)    
    tabletools.saveTable(config['filename_pairs'].replace('.fits','.halos2.fits'), vh2)    
    

def add_phys_dist():

    filename_halos = config['filename_halos']

    big_catalog = tabletools.loadTable(filename_halos,log=logger)

    logger.info('getting cartesian coords')
    # box_coords = cosmology.get_euclidian_coords(big_catalog['ra'],big_catalog['dec'],big_catalog['z'])
    x,y,z = cosmology.spherical_to_cartesian_with_redshift(big_catalog['ra'],big_catalog['dec'],big_catalog['z'])
    DA = cosmology.get_ang_diam_dist(big_catalog['z'])  
    
    big_catalog=tabletools.ensureColumn(rec=big_catalog, name='xphys', arr=x,  dtype='f8')
    big_catalog=tabletools.ensureColumn(rec=big_catalog, name='yphys', arr=y,  dtype='f8')
    big_catalog=tabletools.ensureColumn(rec=big_catalog, name='zphys', arr=z,  dtype='f8')
    big_catalog=tabletools.ensureColumn(rec=big_catalog, name='DA',    arr=DA, dtype='f8')
  
    logger.info('number of halos %d', len(big_catalog))
    tabletools.saveTable(filename_halos,big_catalog)
    logger.info('wrote %s' % filename_halos)

def get_shears_for_pairs(function_shears_for_single_pair,id_first=0, num=-1):
    
    filename_pairs = config['filename_pairs']
    filename_shears = config['filename_shears']
    
    if os.path.isfile(filename_shears): 
        os.remove(filename_shears)
        logger.warning('overwriting file %s' , filename_shears)
        
    halo_pairs = tabletools.loadTable(filename_pairs)
    halo_pairs1 = tabletools.loadTable(filename_pairs.replace('fits','halos1.fits'))    
    halo_pairs2 = tabletools.loadTable(filename_pairs.replace('fits','halos2.fits'))    

    id_last = len(halo_pairs) if num == -1 else id_first + num
    if id_first > len(halo_pairs):
        raise Exception('first pair greater than len(halo_pairs) %d > %d' % (first,len(halo_pairs)))
        
    n_pairs = len(range(id_first,id_last))

    logger.info('getting shears for %d pairs from %d to %d' % (n_pairs , id_first, id_last))

    for ipair in range(id_first,id_last):

        vpair = halo_pairs[ipair]



        # filename_current_pair = filename_shears.replace('.fits', '.%04d.fits' % (ipair))
        filename_current_pair = filename_shears.split('.')
        filename_current_pair[-1]='%04d.fits' % (ipair)
        filename_current_pair = '.'.join(filename_current_pair)
        filename_fig = 'figs/' +  filename_current_pair.replace('.fits','.png')

        halo1 = halo_pairs1[ipair]
        halo2 = halo_pairs2[ipair]
        logger.info('=========== % 4d pair ra=[%6.3f %6.3f] dec=[%6.3f %6.3f] ===========' % (ipair , halo1['ra'] , halo2['ra'] , halo1['dec'] , halo2['dec']))

        pair_shears , halos_coords , pair_shears_full = function_shears_for_single_pair(halo1,halo2,idp=ipair)
        if pair_shears == None:
            log.error('pair_shears == None , something is wrong')
            continue

        halo_pairs['u1_mpc'][ipair] = halos_coords['halo1_u_rot_mpc']
        halo_pairs['v1_mpc'][ipair] = halos_coords['halo1_v_rot_mpc']
        halo_pairs['u2_mpc'][ipair] = halos_coords['halo2_u_rot_mpc']
        halo_pairs['v2_mpc'][ipair] = halos_coords['halo2_v_rot_mpc']
        halo_pairs['u1_arcmin'][ipair] = halos_coords['halo1_u_rot_arcmin']
        halo_pairs['v1_arcmin'][ipair] = halos_coords['halo1_v_rot_arcmin']
        halo_pairs['u2_arcmin'][ipair] = halos_coords['halo2_u_rot_arcmin']
        halo_pairs['v2_arcmin'][ipair] = halos_coords['halo2_v_rot_arcmin']
        halo_pairs['nu'][ipair] = halos_coords['nu']
        halo_pairs['nv'][ipair] = halos_coords['nv']

      
        if config['shear_type'] == 'stacked':

            halo_pairs['n_gal'][ipair] = np.sum(pair_shears['n_gals'])
            n_gals_total=np.sum(pair_shears['n_gals'])
            area=(pair_shears['v_arcmin'].max() - pair_shears['v_arcmin'].min())*(pair_shears['u_arcmin'].max() - pair_shears['u_arcmin'].min())
            halo_pairs['n_gal'][ipair] = n_gals_total
            halo_pairs['area_arcmin2'][ipair] = area
            halo_pairs['n_eff'][ipair] = float(n_gals_total)/area

            if '.fits' in filename_shears:
                tabletools.saveTable(filename_shears,pair_shears,append=True)          
            elif '.pp2' in filename_shears:
                tabletools.savePickle(filename_shears,pair_shears,append=True)          

            if config['save_pairs_plots']:

                pl.figure()
                plot_pair(halos_coords['halo1_u_rot_mpc'], halos_coords['halo1_v_rot_mpc'], halos_coords['halo2_u_rot_mpc'],halos_coords['halo2_v_rot_mpc'], pair_shears['u_mpc'], pair_shears['v_mpc'], pair_shears['g1'], pair_shears['g2'],idp=ipair,filename_fig=filename_fig,halo1=halo1,halo2=halo2,pair_info=vpair,quiver_scale=2)      
                pl.close()

        elif config['shear_type'] == 'single':

            halo_pairs['n_gal'][ipair] = len(pair_shears)

            tabletools.saveTable(filename_current_pair,pair_shears)
    
            if config['save_pairs_plots']:
                
                pl.figure()
                plot_pair(halos_coords['halo1_u_rot_mpc'], halos_coords['halo1_v_rot_mpc'], halos_coords['halo2_u_rot_mpc'], halos_coords['halo2_v_rot_mpc'], pair_shears['u_mpc'], pair_shears['v_mpc'], pair_shears['g1'], pair_shears['g2'],idp=ipair,filename_fig=filename_fig,halo1=halo1,halo2=halo2,pair_info=vpair)
                pl.close()

        elif config['shear_type'] == 'minimal':

            halo_pairs['n_gal'][ipair] = np.sum(pair_shears['n_gals'])

            tabletools.saveTable(filename_current_pair,pair_shears)

            if config['save_pairs_plots']:
                
                pl.figure()
                plot_pair(halos_coords['halo1_u_rot_mpc'], halos_coords['halo1_v_rot_mpc'], halos_coords['halo2_u_rot_mpc'], halos_coords['halo2_v_rot_mpc'], pair_shears_full['u_mpc'], pair_shears_full['v_mpc'], pair_shears_full['g1'], pair_shears_full['g2'],idp=ipair,filename_fig=filename_fig,halo1=halo1,halo2=halo2,pair_info=vpair)
                pl.close()
    
        else: raise ValueError('wrong shear type in config: %s' % config['shear_type'])

        logger.info('%5d ra=% 2.4f (% 2.4f, % 2.4f)  dec=% 2.4f (% 2.4f, % 2.4f) z=% 2.4f (% 2.4f, % 2.4f) R_pair=% 10.4f n_selected=%6d' % (
            ipair,vpair['ra_mid'],halo1['ra'],halo2['ra'],vpair['dec_mid'],halo1['dec'],halo2['dec'],vpair['z'],halo1['z'],halo2['z'], vpair['R_pair'], len(pair_shears)))     

        tabletools.saveTable(filename_pairs,halo_pairs)

