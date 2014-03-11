import pyfits, os, yaml, argparse, sys, logging , cosmology , plotstools , tabletools
import numpy as np
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import warnings; warnings.simplefilter('once')

cospars = cosmology.cosmoparams()

# Dxy = R_pair and drloss = Dlos
dtype_pairs = { 'names'   : ['ipair','ih1','ih2','n_gal','DA','Dlos','Dxy','ra_mid','dec_mid','z', 'ra1','dec1','ra2','dec2','u1_mpc','v1_mpc' , 'u2_mpc','v2_mpc' ,'u1_arcmin','v1_arcmin', 'u2_arcmin','v2_arcmin', 'R_pair','drloss','dz'] ,
                'formats' : ['i8']*4 + ['f8']*21 }

dtype_shears_stacked = { 'names' : ['u_mpc','v_mpc','u_arcmin','v_arcmin','g1','g2','mean_scinv', 'g1sc','g2sc','weight','n_gals'] , 'formats' : ['f8']*10 + ['i8']*1 }
dtype_shears_single = { 'names' : ['ra_deg','dec_deg','u_mpc','v_mpc','g1','g2', 'g1_orig','g2_orig','scinv','z'] , 'formats' : ['f4']*10 }
dtype_shears_minimal = { 'names' : ['ra_deg','dec_deg','g1_orig','g2_orig','scinv','z'] , 'formats' : ['f4']*6 }

# logging.basicConfig(level=logging.INFO,format='%(message)s')
# logger = logging.getLogger("filaments_tools") 

logger = logging.getLogger("fil..tools") 
logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False

config = {}

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


def create_filament_stamp(halo1_ra_deg,halo1_de_deg,halo2_ra_deg,halo2_de_deg,shear_ra_deg,shear_de_deg,shear_g1,shear_g2,shear_z,pair_z,lenscat=None):

        # import pylab as pl
        # pl.figure()
        # pl.scatter( 0     , 0 , 100, 'r' , marker='x')
        # pl.scatter(halo1_ra_deg     , halo1_de_deg , 100, 'c' , marker='o')
        # pl.scatter(halo2_ra_deg     , halo2_de_deg ,100,  'm' , marker='o')
        # select = np.random.permutation(len(shear_ra_deg))[:10000]
        # pl.scatter(shear_ra_deg[select]  , shear_de_deg[select] ,1,  'm' , marker='.')
        # pl.show()


        # convert to radians
        halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(halo1_ra_deg, halo1_de_deg)
        halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(halo2_ra_deg, halo2_de_deg)
        shear_ra_rad , shear_de_rad = cosmology.deg_to_rad(shear_ra_deg, shear_de_deg)

        # get midpoint
        pairs_ra_rad , pairs_de_rad = cosmology.get_midpoint_rad(halo1_ra_rad,halo1_de_rad,halo2_ra_rad,halo2_de_rad)

        # # localise
        halo1_u_rad , halo1_v_rad = halo1_ra_rad - pairs_ra_rad , halo1_de_rad - pairs_de_rad
        halo2_u_rad , halo2_v_rad = halo2_ra_rad - pairs_ra_rad , halo2_de_rad - pairs_de_rad
        shear_u_rad , shear_v_rad = shear_ra_rad - pairs_ra_rad , shear_de_rad - pairs_de_rad 
        pairs_u_rad , pairs_v_rad =  0, 0

        # then wrap
        pairs_u_rad_wrapped , pairs_v_rad_wrapped = wrap_angle_rad(pairs_u_rad) , wrap_angle_rad(pairs_v_rad)
        halo1_u_rad_wrapped , halo1_v_rad_wrapped = wrap_angle_rad(halo1_u_rad) , wrap_angle_rad(halo1_v_rad)
        halo2_u_rad_wrapped , halo2_v_rad_wrapped = wrap_angle_rad(halo2_u_rad) , wrap_angle_rad(halo2_v_rad)
        shear_u_rad_wrapped , shear_v_rad_wrapped = wrap_angle_rad(shear_u_rad) , wrap_angle_rad(shear_v_rad)
        
        # import pylab as pl
        # pl.figure()
        # pl.scatter( 0     , 0 , 100, 'r' , marker='x')
        # pl.scatter(halo1_u_rad_wrapped     , halo1_v_rad_wrapped , 100, 'c' , marker='o')
        # pl.scatter(halo2_u_rad_wrapped     , halo2_v_rad_wrapped ,100,  'm' , marker='o')
        # pl.scatter( min(shear_u_rad_wrapped) , min(shear_v_rad_wrapped) , 50 , marker='d')
        # pl.scatter( max(shear_u_rad_wrapped) , min(shear_v_rad_wrapped) , 50 , marker='d')
        # pl.scatter( min(shear_u_rad_wrapped) , max(shear_v_rad_wrapped) , 50 , marker='d')
        # pl.scatter( max(shear_u_rad_wrapped) , max(shear_v_rad_wrapped) , 50 , marker='d')
        # pl.show()

        
        # linearise
        halo1_u_rad = halo1_u_rad_wrapped * np.cos(pairs_v_rad_wrapped)
        halo2_u_rad = halo2_u_rad_wrapped * np.cos(halo2_v_rad_wrapped)
        shear_u_rad = shear_u_rad_wrapped * np.cos(shear_v_rad_wrapped)
        # # possibly shear rotation too here?

        # import pylab as pl
        # pl.figure()
        # select = np.random.permutation(len(shear_u_rad))[:10000]
        # pl.scatter( shear_u_rad[select] , shear_v_rad[select] , 1 , marker='.')
        # pl.scatter( 0     , 0 , 100, 'r' , marker='x')
        # pl.scatter(halo1_u_rad , halo1_v_rad , 100, 'c' , marker='o')
        # pl.scatter(halo2_u_rad , halo2_v_rad , 100, 'm' , marker='o')
        # pl.show()

        rotation_angle = np.angle(halo1_u_rad + 1j*halo1_v_rad)

        shear_u_rot_rad , shear_v_rot_rad = rotate_vector(rotation_angle, shear_u_rad , shear_v_rad)
        halo1_u_rot_rad , halo1_v_rot_rad = rotate_vector(rotation_angle, halo1_u_rad , halo1_v_rad)
        halo2_u_rot_rad , halo2_v_rot_rad = rotate_vector(rotation_angle, halo2_u_rad , halo2_v_rad)   
        shear_g1_rot , shear_g2_rot = rotate_shear(rotation_angle, shear_u_rad, shear_v_rad, shear_g1, shear_g2)

        # import pylab as pl
        # pl.figure()
        # select = np.random.permutation(len(shear_v_rot_rad))[:10000]
        # pl.scatter( shear_u_rot_rad[select] ,shear_v_rot_rad[select] , 1 , marker=',')
        # pl.scatter( 0     , 0 , 100, 'r' , marker='x')
        # pl.scatter(halo1_u_rot_rad , halo1_v_rot_rad , 100, 'c' , marker='o')
        # pl.scatter(halo2_u_rot_rad , halo2_v_rot_rad , 100, 'm' , marker='o')
        # pl.show()

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

            # import pylab as pl
            # pl.figure()
            # select = np.random.permutation(len(shear_u_rot_rad))[:10000]
            # pl.scatter( shear_u_rot_rad[select] ,shear_v_rot_rad[select] , 1 , marker=',')
            # pl.scatter( 0     , 0 , 100, 'r' , marker='x')
            # pl.scatter(halo1_u_rot_rad , halo1_v_rot_rad , 100, 'c' , marker='o')
            # pl.scatter(halo2_u_rot_rad , halo2_v_rot_rad , 100, 'm' , marker='o')
            # pl.scatter(  np.abs(halo1_u_rot_rad + dtheta_x) ,  np.abs(dtheta_y) , marker='+' )
            # pl.scatter( -np.abs(halo1_u_rot_rad + dtheta_x) ,  np.abs(dtheta_y) , marker='+' )
            # pl.scatter(  np.abs(halo1_u_rot_rad + dtheta_x) , -np.abs(dtheta_y) , marker='+' )
            # pl.scatter( -np.abs(halo1_u_rot_rad + dtheta_x) , -np.abs(dtheta_y) , marker='+' )
            # pl.show()

            # select = (shear_v_rot_rad < dtheta_y) * (shear_v_rot_rad > -dtheta_y) * (shear_u_rot_rad < (halo1_u_rot_rad + dtheta_x)) *  (shear_u_rot_rad > (halo2_u_rot_rad - dtheta_x))
            select = ( np.abs( shear_u_rot_rad ) < np.abs(halo1_u_rot_rad + dtheta_x)) * (np.abs(shear_v_rot_rad) < np.abs(dtheta_y))

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

        # import pylab as pl
        # pl.figure()
        # select = np.random.permutation(len(shear_u_stamp_rad))[:10000]
        # pl.scatter( shear_u_stamp_rad[select] ,shear_v_stamp_rad[select] , 1 , marker=',')
        # pl.scatter( 0     , 0 , 100, 'r' , marker='x')
        # pl.scatter(halo1_u_rot_rad , halo1_v_rot_rad , 100, 'c' , marker='o')
        # pl.scatter(halo2_u_rot_rad , halo2_v_rot_rad , 100, 'm' , marker='o')
        # pl.show()

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

        logger.info('r_pair = %2.2f Mpc = %2.2f arcmin ' , np.abs(halo1_u_rot_mpc - halo2_u_rot_mpc) , np.abs(halo1_u_rot_arcmin - halo2_u_rot_arcmin))


        if config['shear_type'] == 'stacked':
           
            pixel_size_arcmin,_ = cosmology.mpc_to_arcmin(config['pixel_size_mpc'],0.,pair_z) 

            grid_u_mpc = np.arange( -range_u_mpc / 2. , range_u_mpc / 2., config['pixel_size_mpc'] )
            grid_v_mpc = np.arange( -range_v_mpc / 2. , range_v_mpc / 2., config['pixel_size_mpc'] )
            grid_u_arcmin = grid_u_mpc / config['pixel_size_mpc'] * pixel_size_arcmin 
            grid_v_arcmin = grid_v_mpc / config['pixel_size_mpc'] * pixel_size_arcmin


            logger.info('len grid_u_mpc=%d grid_v_mpc=%d' , len(grid_u_mpc) ,  len(grid_v_mpc))
            
            hist_g1, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_g1_stamp)
            hist_g2, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=shear_g2_stamp)
            hist_g1sc, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=g1sc)
            hist_g2sc, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=g2sc)
            hist_n,  _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) )
            hist_scinv, _, _ = np.histogram2d( x=shear_u_stamp_mpc, y=shear_v_stamp_mpc , bins=(grid_u_mpc,grid_v_mpc) , weights=1./sc )

            mean_g1 = hist_g1 / hist_n
            mean_g2 = hist_g2 / hist_n
            mean_scinv = hist_scinv / hist_n
            mean_g1sc = hist_g1sc / hist_n
            mean_g2sc = hist_g2sc / hist_n

            # in case we divided by zero
            mean_g1[hist_n == 0] = 0
            mean_g2[hist_n == 0] = 0

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
            weight = g1*0 + 1. # set all rows to 1 
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

            # dtype_shears_stacked = { 'names' : ['u_mpc','v_mpc','u_arcmin','v_arcmin','g1sc','g2sc','sc_sum',',weight','n_gals'] , 'formats' : ['f8']*8 + ['i8']*1 }
            pairs_shear = np.concatenate([u_mpc,v_mpc,u_arcmin,v_arcmin,g1,g2,mean_scinv,weight,g1sc,g2sc,n_gals],axis=1)
            pairs_shear = tabletools.array2recarray(pairs_shear,dtype_shears_stacked)        


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
            weight = ra*0 + 1. # set all rows to 1 
            n_gals = ra*0 + 1. # set all rows to 1
            scinv = scinv[:,None]
            z = shear_z_stamp[:,None]

            pairs_shear = np.concatenate([ra,de,u_mpc,v_mpc,g1,g2,g1_orig,g2_orig,scinv,z],axis=1)
            pairs_shear = tabletools.array2recarray(pairs_shear,dtype_shears_single)        

        elif config['shear_type'] == 'minimal':

            ra = lenscat_stamp['ra'][:,None]
            de = lenscat_stamp['dec'][:,None]
            g1_orig = shear_g1_orig[:,None]
            g2_orig = shear_g2_orig[:,None]
            scinv = scinv[:,None]
            z = shear_z_stamp[:,None]

            pairs_shear = np.concatenate([ra,de,g1_orig,g2_orig,scinv,z],axis=1)
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

        return pairs_shear , halos_coords


def linearise_coords(ra_deg,de_deg,center_ra_deg,center_de_deg):

    
    ra_rad , de_rad = cosmology.deg_to_rad(ra_deg,de_deg)
    center_ra_rad , center_de_rad = cosmology.deg_to_rad(center_ra_deg,center_de_deg)
    u_rad  , v_rad  = cosmology.get_uv_coords(ra_rad,de_rad,center_ra_rad,center_de_rad)

    # logger.info('ra=%f de=%f (deg) ra=%f de=%f (rad) u=%f v=%f'%(ra_deg,de_deg,ra_rad,de_rad,u_rad,v_rad))

    return u_rad, v_rad 

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

def plot_pair(halo1_x,halo1_y,halo2_x,halo2_y,shear_x,shear_y,shear_g1,shear_g2,idp=0,nuse=10,filename_fig=None,show=False,close=True,halo1=None,halo2=None,pair_info=None,quiver_scale=2):
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

    pl.quiver(shear_x[::nuse],shear_y[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.005,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale)
    pl.scatter(halo1_x,halo1_y,100,c='b') 
    pl.scatter(halo2_x,halo2_y,100,c='c') 

    if (halo1 != None) and (halo2 != None):
        pl.title('%s r_pair=%2.2fMpc n_gals=%d M1=%2.2e M2=%2.2e' % (idp,r_pair,len(shear_g1),halo1['m200'],halo2['m200']))
    else:
        pl.title('%s r_pair=%2.2fMpc n_gals=%d' % (idp,r_pair,len(shear_g1)))
    
    pl.xlim([min(shear_x),max(shear_x)])
    pl.ylim([min(shear_y),max(shear_y)])
    pl.axis('equal')

    if filename_fig != None:
        pl.savefig(filename_fig)
        logger.info('saved %s' % filename_fig)
    if show:
        pl.show()
    if close:
        pl.close()



def stats_pairs(filename_pairs):

    import pylab as pl

    pairs_table = tabletools.loadTable(filename_pairs)

    logger.info('getting stats')
    
    pl.figure()
    logger.info('distribution of los distance between halos Mpc')
    pl.hist(pairs_table['Dlos'],bins=range(2,20))
    pl.title('distribution of los distance between halos Mpc')
    pl.xlabel('Rlos')
    filename_fig = 'los_pairs.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' % filename_fig)
    pl.close()

    # otherwise it takes forever

    pl.figure()
    logger.info('distribution of length of connections')
    pl.hist(pairs_table['R_pair'],bins=range(2,20))
    pl.title('distribution of length of connections Mpc')
    pl.xlabel('Rpair')
    filename_fig = 'r_pairs.png'
    pl.savefig(filename_fig)
    logger.info('saved %s' % filename_fig)
    pl.close()

    if len(pairs_table) < 1000:
        logger.info('distribution of number of connections per cluster')
        counts = np.array([ sum(pairs_table['ih1']==ih1) for ih1 in pairs_table['ih1']])
        pl.hist(counts,bins=range(0,8))
        pl.title('distribution of number of connections per cluster')
        pl.xlabel('n connections')
        filename_fig = 'n_connections_per_halo.png'
        pl.savefig(filename_fig)
        pl.close()
        logger.info('saved %s' % filename_fig)



def get_pairs(range_Dxy=[6,18],Dlos=6,filename_halos='big_halos.fits'):

    logger.info('%s' % str(cospars))
    
    halocat = tabletools.loadTable(filename_halos)
    logger.info('number of halos: %d' % len(halocat))  

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
     
    logger.info(str(sum(select)))

    logger.info('number of pairs %d ' % len(ih1))
    logger.info('removing duplicates')
    select = ih1 < ih2
    ih1 = ih1[select]
    ih2 = ih2[select]
    DA = DA[select]
    logger.info('number of pairs %d ' % len(ih1))

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
    logger.info('select the plane separation')
    select1 = np.abs(d_los) < max_los
    select2 = d_xy > range_Dxy[0]
    select3 = d_xy < range_Dxy[1]

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

    ipair = np.ones(len(ih1),dtype=np.int8)
    # 'names'   : ['ipair','ih1','ih2','n_gal','DA','Dlos','Dxy','ra_mid','dec_mid','z', 'ra1','dec1','ra2','dec2','u1_mpc','v1_mpc' , 'u2_mpc','v2_mpc' ,'u1_arcmin','v1_arcmin', 'u2_arcmin','v2_arcmin', 'R_pair','drloss','dz'] 
    row = [ipair[:,None],ih1[:,None],ih2[:,None],n_gal[:,None],
            DA[:,None],d_los[:,None],d_xy[:,None],
            ra_mid[:,None],dec_mid[:,None],z[:,None],
            vh1['ra'][:,None],vh1['dec'][:,None],vh2['ra'][:,None],vh2['dec'][:,None],
            ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0,ipair[:,None]*0, # these fields will be filled in later
            R_pair[:,None],drloss[:,None],dz[:,None]]
    pairs_table = np.concatenate(row,axis=1)
    pairs_table = tabletools.array2recarray(pairs_table,dtype_pairs)
    
    return (pairs_table, vh1, vh2)

def add_phys_dist(filename_halos):

    big_catalog = tabletools.loadTable(filename_halos,log=logger)

    logger.info('getting cartesian coords')
    # box_coords = cosmology.get_euclidian_coords(big_catalog['ra'],big_catalog['dec'],big_catalog['z'])
    x,y,z = cosmology.spherical_to_cartesian_with_redshift(big_catalog['ra'],big_catalog['dec'],big_catalog['z'])
    DA = cosmology.get_ang_diam_dist(big_catalog['z'])  
    if 'xphys' not in big_catalog.dtype.names:
        logger.info('adding new columns')
        big_catalog=tabletools.appendColumn(big_catalog, 'xphys', x, dtype='f8')
        big_catalog=tabletools.appendColumn(big_catalog, 'yphys', y, dtype='f8')
        big_catalog=tabletools.appendColumn(big_catalog, 'zphys', z, dtype='f8')
        big_catalog=tabletools.appendColumn(big_catalog, 'DA', DA, dtype='f8')
    else:
        logger.info('updating columns')
        big_catalog['xphys'] = x
        big_catalog['yphys'] = y
        big_catalog['zphys'] = z
        big_catalog['DA'] = DA

    logger.info('number of halos %d', len(big_catalog))
    tabletools.saveTable(filename_halos,big_catalog)
    logger.info('wrote %s' % filename_halos)

def get_shears_for_pairs(filename_pairs, filename_shears, function_shears_for_single_pair, filename_full_halocat = None , id_first=0, id_last=-1):

    if os.path.isfile(filename_shears): raise Exception('file %s already exists' % filename_shears)

    halo_pairs = tabletools.loadTable(filename_pairs)
    halo_pairs1 = tabletools.loadTable(filename_pairs.replace('fits','halos1.fits'))    
    halo_pairs2 = tabletools.loadTable(filename_pairs.replace('fits','halos2.fits'))    
    if filename_full_halocat != None:
        full_halocat = tabletools.loadTable(filename_full_halocat)    

    list_fitstb = []

    if id_last == -1:
        id_last = len(halo_pairs) 

    n_pairs = len(range(id_first,id_last))

    logger.info('getting shears for %d pairs' % n_pairs)

    for ipair in range(id_first,id_last):

        vpair = halo_pairs[ipair]



        filename_current_pair = filename_shears.replace('.fits', '.%03d.fits' % (ipair))
        filename_fig =  filename_current_pair.replace('.fits','.png')

        halo1 = halo_pairs1[ipair]
        halo2 = halo_pairs2[ipair]
        logger.info('=========== % 4d pair ra=[%6.3f %6.3f] dec=[%6.3f %6.3f] ===========' % (ipair , halo1['ra'] , halo2['ra'] , halo1['dec'] , halo2['dec']))

        pair_shears , halos_coords = function_shears_for_single_pair(halo1,halo2,idp=ipair)
        if pair_shears == None:
            log.error('pair_shears == None , something is wrong')
            continue

        halo_pairs['n_gal'][ipair] = len(pair_shears)
        halo_pairs['u1_mpc'][ipair] = halos_coords['halo1_u_rot_mpc']
        halo_pairs['v1_mpc'][ipair] = halos_coords['halo1_v_rot_mpc']
        halo_pairs['u2_mpc'][ipair] = halos_coords['halo2_u_rot_mpc']
        halo_pairs['v2_mpc'][ipair] = halos_coords['halo2_v_rot_mpc']
        halo_pairs['u1_arcmin'][ipair] = halos_coords['halo1_u_rot_arcmin']
        halo_pairs['v1_arcmin'][ipair] = halos_coords['halo1_v_rot_arcmin']
        halo_pairs['u2_arcmin'][ipair] = halos_coords['halo2_u_rot_arcmin']
        halo_pairs['v2_arcmin'][ipair] = halos_coords['halo2_v_rot_arcmin']

      
        if config['shear_type'] == 'stacked':

            if config['save_pairs_plots']:
                # plot_pair(halos_coords['halo1_u_rot_mpc'] , halos_coords['halo1_v_rot_mpc'] , halos_coords['halo2_u_rot_mpc'] , halos_coords['halo2_v_rot_mpc'] , u_mpc, v_mpc, g1sc, g2sc , close=False,nuse = 1,quiver_scale=15)
                plot_pair(halos_coords['halo1_u_rot_mpc'], halos_coords['halo1_v_rot_mpc'], halos_coords['halo2_u_rot_mpc'],halos_coords['halo2_v_rot_mpc'], pair_shears['u_mpc'], pair_shears['v_mpc'], pair_shears['g1'], pair_shears['g2'],idp=ipair,filename_fig=filename_fig,halo1=halo1,halo2=halo2,pair_info=vpair,quiver_scale=2,nuse=1)
            
            tabletools.saveTable(filename_shears,pair_shears,append=True)          
       
        elif config['shear_type'] == 'single':
    
            if config['save_pairs_plots']:
                # plot_pair(halo1,halo2,vpair,halo1['ra'], halo1['dec'], halo2['ra'], halo2['dec'], pair_shears['ra_deg'], pair_shears['dec_deg'], pair_shears['g1_orig'], pair_shears['g2_orig'],idp=ipair,filename_fig=filename_fig)
                plot_pair(halos_coords['halo1_u_rot_mpc'], halos_coords['halo1_v_rot_mpc'], halos_coords['halo2_u_rot_mpc'], halos_coords['halo2_v_rot_mpc'], pair_shears['u_mpc'], pair_shears['v_mpc'], pair_shears['g1'], pair_shears['g2'],idp=ipair,filename_fig=filename_fig,halo1=halo1,halo2=halo2,pair_info=vpair)
    
            tabletools.saveTable(filename_current_pair,pair_shears)
            
        elif config['shear_type'] == 'minimal':

            tabletools.saveTable(filename_current_pair,pair_shears)
            logger.info('saved %s' % filename_current_pair)

            # plot_pair(halos_coords['halo1_u_rot_mpc'], halos_coords['halo1_v_rot_mpc'], halos_coords['halo2_u_rot_mpc'], halos_coords['halo2_v_rot_mpc'], pair_shears['u_mpc'], pair_shears['v_mpc'], pair_shears['g1'], pair_shears['g2'],idp=ipair,filename_fig=filename_fig,halo1=halo1,halo2=halo2,pair_info=vpair)
    
        else: raise ValueError('wrong shear type in config: %s' % config['shear_type'])

        logger.info('%5d ra=% 2.4f (% 2.4f, % 2.4f)  dec=% 2.4f (% 2.4f, % 2.4f) z=% 2.4f (% 2.4f, % 2.4f) R_pair=% 10.4f n_selected=%6d' % (
            ipair,vpair['ra_mid'],halo1['ra'],halo2['ra'],vpair['dec_mid'],halo1['dec'],halo2['dec'],vpair['z'],halo1['z'],halo2['z'], vpair['R_pair'], len(pair_shears)))     

        tabletools.saveTable(filename_pairs,halo_pairs)
        logger.info( 'wrote %s' % filename_pairs )    

