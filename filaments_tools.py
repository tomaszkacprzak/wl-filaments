import pyfits
import numpy as np
import pylab as pl
import scipy.interpolate as interp
import cosmology
import tabletools
import yaml, argparse, sys, logging 
from sklearn.neighbors import BallTree as BallTree
import galsim

cospars = cosmology.cosmoparams()

# Dxy = R_pair and drloss = Dlos
dtype_pairs = { 'names'   : ['ipair','ih1','ih2','n_gal','DA','Dlos','Dxy','ra_mid','dec_mid','z', 'ra1','dec1','ra2','dec2','u1_mpc','v1_mpc' , 'u2_mpc','v2_mpc' ,'u1_arcmin','v1_arcmin', 'u2_arcmin','v2_arcmin', 'R_pair','drloss','dz'] ,
                'formats' : ['i8']*4 + ['f8']*21 }

dtype_shears = { 'names' : ['u_mpc','v_mpc','u_arcmin','v_arcmin','ra_deg','dec_deg','g1','g2','scinv','weight','z','n_gals'] , 'formats' : ['f8']*11 + ['i8']*1 }
dtype_shears_reduced = { 'names' : ['ra_deg','dec_deg','g1','g2','g1_orig','g2_orig','scinv'] , 'formats' : ['f8']*7 }

logging.basicConfig(level=logging.DEBUG,format='%(message)s')
logger = logging.getLogger("filaments_tools") 
logger.setLevel(logging.INFO)

boundary_mpc=4
tag='g'

def get_galaxy_density(shear_ra_deg,shear_de_deg):

    shear_ra_arcmin , shear_de_arcmin = cosmology.deg_to_arcmin(shear_ra_deg , shear_de_deg)

    dra = max(shear_ra_arcmin) - min(shear_ra_arcmin)
    dde = max(shear_de_arcmin) - min(shear_de_arcmin)

    area = dra*dde
    density = float(len(shear_ra_arcmin)) / area
    return density

def wrap_angle(ang_rad):

    if isinstance(ang_rad,np.ndarray):
        select = ang_rad > np.pi
        ang_rad[select] -= np.pi
    elif ang_rad > np.pi:
        ang_rad -= np.pi

    return ang_rad


def create_filament_stamp(halo1_ra_deg,halo1_de_deg,halo2_ra_deg,halo2_de_deg,shear_ra_deg,shear_de_deg,shear_g1,shear_g2,pair_z,lenscat=None):


        pairs_ra_deg = np.mean([halo1_ra_deg,halo2_ra_deg])
        pairs_de_deg = np.mean([halo1_de_deg,halo2_de_deg])

        # convert to radians
        pairs_ra_rad , pairs_de_rad = cosmology.deg_to_rad(pairs_ra_deg, pairs_de_deg)
        halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(halo1_ra_deg, halo1_de_deg)
        halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(halo2_ra_deg, halo2_de_deg)
        shear_ra_rad , shear_de_rad = cosmology.deg_to_rad(shear_ra_deg, shear_de_deg)
        
        # localise
        halo1_u_rad , halo1_v_rad = halo1_ra_rad - pairs_ra_rad , halo1_de_rad - pairs_de_rad
        halo2_u_rad , halo2_v_rad = halo2_ra_rad - pairs_ra_rad , halo2_de_rad - pairs_de_rad
        shear_u_rad , shear_v_rad = shear_ra_rad - pairs_ra_rad , shear_de_rad - pairs_de_rad 
        halo1_u_rad , halo1_v_rad = wrap_angle(halo1_u_rad) , wrap_angle(halo1_v_rad) # not sure if this works 
        halo2_u_rad , halo2_v_rad = wrap_angle(halo2_u_rad) , wrap_angle(halo2_v_rad) # not sure if this works
        shear_u_rad , shear_v_rad = wrap_angle(shear_u_rad) , wrap_angle(shear_v_rad) # not sure if this works      

        # linearise
        halo1_u_rad = halo1_u_rad * np.cos(pairs_de_rad)
        halo2_u_rad = halo2_u_rad * np.cos(pairs_de_rad)
        shear_u_rad = shear_u_rad * np.cos(pairs_de_rad)

        rotation_angle = np.angle(halo1_u_rad + 1j*halo1_v_rad)

        shear_u_rot_rad , shear_v_rot_rad = rotate_vector(rotation_angle, shear_u_rad , shear_v_rad)
        halo1_u_rot_rad , halo1_v_rot_rad = rotate_vector(rotation_angle, halo1_u_rad , halo1_v_rad)
        halo2_u_rot_rad , halo2_v_rot_rad = rotate_vector(rotation_angle, halo2_u_rad , halo2_v_rad)   
        shear_g1_rot , shear_g2_rot = rotate_shear(rotation_angle, shear_u_rad, shear_v_rad, shear_g1, shear_g2)

        
        # grid boudaries

        # find the angle which corresponds to boundary_mpc at current redshift
        if boundary_mpc == '3x4':
            r_pair = np.abs(halo1_u_rot_rad - halo2_u_rot_rad)
            dtheta_x = r_pair * 1.6
            dtheta_y = r_pair * 2.1
            select = (shear_v_rot_rad < dtheta_y) * (shear_v_rot_rad > -dtheta_y) * (shear_u_rot_rad < dtheta_x) *  (shear_u_rot_rad > -dtheta_x)
        else:
            dtheta_x = boundary_mpc / cosmology.get_ang_diam_dist(pair_z)
            dtheta_y = boundary_mpc / cosmology.get_ang_diam_dist(pair_z)
            select = (shear_v_rot_rad < dtheta_y) * (shear_v_rot_rad > -dtheta_y) * (shear_u_rot_rad < (halo1_u_rot_rad + dtheta_x)) *  (shear_u_rot_rad > (halo2_u_rot_rad - dtheta_x))

        logger.info('z=%f dtheta=%f' % (pair_z,dtheta_x))

        # select the stamp

        shear_u_stamp_rad  = shear_u_rot_rad[select]
        shear_v_stamp_rad  = shear_v_rot_rad[select]
        shear_g1_stamp = shear_g1_rot[select]
        shear_g2_stamp = shear_g2_rot[select]
        
        # logger.info('r_pair=%2.2e rad ' % np.abs(halo1_u_rot_rad - halo2_u_rot_rad))
        # logger.info('dtheta_x=%2.2e rad ' % r_pair)
        # pl.scatter(shear_u_stamp_rad,shear_v_stamp_rad,c='b')
        # pl.scatter(halo1_u_rot_rad,halo1_v_rot_rad,c='r')
        # pl.scatter(halo2_u_rot_rad,halo2_v_rot_rad,c='r')
        # pl.axis('equal')
        # pl.show()




        if lenscat != None:
            lenscat_stamp  = lenscat[select]
        else:
            lenscat_stamp=None

        # convert to Mpc
        shear_u_stamp_mpc , shear_v_stamp_mpc = cosmology.rad_to_mpc(shear_u_stamp_rad,shear_v_stamp_rad,pair_z)
        halo1_u_rot_mpc , halo1_v_rot_mpc = cosmology.rad_to_mpc(halo1_u_rot_rad,halo1_v_rot_rad,pair_z)
        halo2_u_rot_mpc , halo2_v_rot_mpc = cosmology.rad_to_mpc(halo2_u_rot_rad,halo2_v_rot_rad,pair_z)

        shear_u_stamp_arcmin , shear_v_stamp_arcmin = cosmology.rad_to_arcmin(shear_u_stamp_rad,shear_v_stamp_rad)
        halo1_u_rot_arcmin , halo1_v_rot_arcmin = cosmology.rad_to_arcmin(halo1_u_rot_rad,halo1_v_rot_rad)
        halo2_u_rot_arcmin , halo2_v_rot_arcmin = cosmology.rad_to_arcmin(halo2_u_rot_rad,halo2_v_rot_rad)

        logger.info('r_pair=%2.2e Mpc ' % np.abs(halo1_u_rot_mpc - halo2_u_rot_mpc))
        logger.info('r_pair=%2.2e arcmin ' % np.abs(halo1_u_rot_arcmin - halo2_u_rot_arcmin))

        return (halo1_u_rot_mpc,      halo1_v_rot_mpc, 
                halo2_u_rot_mpc,      halo2_v_rot_mpc, 
                halo1_u_rot_arcmin,   halo1_v_rot_arcmin, 
                halo2_u_rot_arcmin,   halo2_v_rot_arcmin, 
                shear_u_stamp_mpc,    shear_v_stamp_mpc, 
                shear_u_stamp_arcmin, shear_v_stamp_arcmin, 
                shear_g1_stamp, shear_g2_stamp, lenscat_stamp)


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

    shear_pos = (shear_ra + 1j*shear_de)*np.exp(-1j*rotation_angle)
    shear_ra_rot = shear_pos.real 
    shear_de_rot = shear_pos.imag
    shear_g = (shear_g1 + 1j*shear_g2)*np.exp(-2j*rotation_angle)
    shear_g1_rot = shear_g.real
    shear_g2_rot = shear_g.imag

    return shear_g1_rot, shear_g2_rot

def plot_pair(halo1,halo2,pair_info,halo1_x,halo1_y,halo2_x,halo2_y,shear_x,shear_y,shear_g1,shear_g2,idp=0,nuse=10,filename_fig='whiskers.png',show=False,close=True):
    """
    In Mpc
    """

    # pl.figure(figsize=(50,25),dpi=500)
    pl.figure()
    pl.clf()
    quiver_scale=2
    r_pair = pair_info['R_pair']
    emag=np.sqrt(shear_g1**2+shear_g2**2)
    ephi=0.5*np.arctan2(shear_g2,shear_g1)              

    pl.quiver(shear_x[::nuse],shear_y[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.005,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale)
    pl.scatter(halo1_x,halo1_y,100,c='b') 
    pl.scatter(halo2_x,halo2_y,100,c='c') 
    pl.title('%s r_pair=%2.2fMpc n_gals=%d M1=%2.2e M2=%2.2e' % (idp,r_pair,len(shear_g1),halo1['m200'],halo2['m200']))
    pl.xlim([min(shear_x),max(shear_x)])
    pl.ylim([min(shear_y),max(shear_y)])
    pl.axis('equal')


    pl.savefig(filename_fig)
    logger.info('saved %s' % filename_fig)
    if show:
        pl.show()
    if close:
        pl.close()


def stats_pairs(filename_pairs):

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
    print ih2
    print DA
 
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

    big_catalog = tabletools.loadTable(filename_halos,logger=logger)

    logger.info('getting euclidian coords')
    box_coords = cosmology.get_euclidian_coords(big_catalog['ra'],big_catalog['dec'],big_catalog['z'])
    DA = cosmology.get_ang_diam_dist(big_catalog['z'])  
    if 'xphys' not in big_catalog.dtype.names:
        logger.info('adding new columns')
        big_catalog=tabletools.appendColumn(big_catalog, 'xphys', box_coords[:,0], dtype='f8')
        big_catalog=tabletools.appendColumn(big_catalog, 'yphys', box_coords[:,1], dtype='f8')
        big_catalog=tabletools.appendColumn(big_catalog, 'zphys', box_coords[:,2], dtype='f8')
        big_catalog=tabletools.appendColumn(big_catalog, 'DA', DA, dtype='f8')
    else:
        logger.info('updating columns')
        big_catalog['xphys'] = box_coords[:,0]
        big_catalog['yphys'] = box_coords[:,1]
        big_catalog['zphys'] = box_coords[:,2]
        big_catalog['DA'] = DA

    logger.info('number of halos %d', len(big_catalog))
    tabletools.saveTable(filename_halos,big_catalog)
    logger.info('wrote %s' % filename_halos)

def get_shears_for_pairs(filename_pairs, filename_shears, function_shears_for_single_pair, n_pairs=100, filename_full_halocat = None):

    halo_pairs = tabletools.loadTable(filename_pairs)
    halo_pairs1 = tabletools.loadTable(filename_pairs.replace('fits','halos1.fits'))    
    halo_pairs2 = tabletools.loadTable(filename_pairs.replace('fits','halos2.fits'))    
    if filename_full_halocat != None:
        full_halocat = tabletools.loadTable(filename_full_halocat)    

    list_fitstb = []

    logger.info('getting shears for %d pairs' % n_pairs)

    for ipair,vpair in enumerate(halo_pairs[:n_pairs]):

        filename_current_pair = '%s.%03d.fits' % (filename_shears,ipair)

        halo1 = halo_pairs1[ipair]
        halo2 = halo_pairs2[ipair]
        
        pair_shears , halos_coords = function_shears_for_single_pair(halo1,halo2,idp=ipair)
        if pair_shears == None:
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

        filename_fig = '%s.%s.%03d.png' % (filename_shears,tag,ipair)

        plot_pair(halo1,halo2,vpair,halos_coords['halo1_u_rot_mpc'], halos_coords['halo1_v_rot_mpc'], halos_coords['halo2_u_rot_mpc'], halos_coords['halo2_v_rot_mpc'], pair_shears['u_mpc'], pair_shears['v_mpc'], pair_shears['g1'], pair_shears['g2'],idp=ipair,filename_fig=filename_fig)

        gal_density = get_galaxy_density(pair_shears['ra_deg'],pair_shears['dec_deg'])
        logger.info('gal_density %2.2f' % gal_density)

        # list_fitstb.append( tabletools.getBinaryTable(pair_shears) )
        tabletools.saveTable(filename_current_pair,pair_shears)
        logger.info('saved %s' % filename_current_pair)

        logger.info('%5d ra=% 2.4f (% 2.4f, % 2.4f)  dec=% 2.4f (% 2.4f, % 2.4f) z=% 2.4f (% 2.4f, % 2.4f) n_selected=%6d' % (
            ipair,vpair['ra_mid'],halo1['ra'],halo2['ra'],vpair['dec_mid'],halo1['dec'],halo2['dec'],vpair['z'],halo1['z'],halo2['z'], len(pair_shears)))     

    # hdu = pyfits.PrimaryHDU()
    # hdulist = pyfits.HDUList( [hdu] + list_fitstb )
    # hdulist.writeto(filename_shears,clobber=True)                      
    # logger.info( 'wrote %s' % filename_shears )

    tabletools.saveTable(filename_pairs,halo_pairs)
    logger.info( 'wrote %s' % filename_pairs )    
