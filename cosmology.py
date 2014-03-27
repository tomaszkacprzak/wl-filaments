import logging
import numpy as np
import scipy.interpolate as interp

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('cosmology')
logger.setLevel(logging.INFO)

class cosmoparams:

    def __init__(self):

        self.Omega_m = 0.3
        self.Omega_Lambda = 0.7
        self.w = -1.
        self.Omega_k = 0.
        self.h = 0.7
        self.H_0 = 100. # 
        self.DH= 3000.       # Mpc
        self.G = 6.673e-11; # SI units m3 kg-1 s-2
        self.c =  299792458; # m/s
        self.Mpc_to_m = 3.0856e22; # number of meters in 1Mpc
        self.M_solar = 1.989e30; # kg / M_solar
        self.H_0__h_over_s = self.H_0 * 1e3 / self.Mpc_to_m  # h/s
        self.rho_crit = 3 * self.H_0__h_over_s**2 / (8 *np.pi * self.G) /self.M_solar *self.Mpc_to_m**3; # h^2 M_solar / Mpc^-3
        print 'cosmology: rho_crit', self.rho_crit
        # G = 6.672e-8;
        # C = 29979245800;
        # Pc = 3.0857e18;

cospars = cosmoparams()


def get_comoving_dist_line_of_sight(z):
    """
    @return in Mpc
    """
    if isinstance(z,float):
        if z==0.:
            return 0.

    n_grid_z = 1000
    try:
        maxz = max(z)
    except Exception, te:
        # print 'TypeError',  te, z
        maxz = z
       


    if maxz < 0.001:
        print 'small z' , maxz , type(maxz), z
        import pdb;pdb.set_trace()

    grid_z = np.linspace(0,maxz,n_grid_z)
    dz = grid_z[1]
    E = np.sqrt(cospars.Omega_m * (1+grid_z)**3 + cospars.Omega_k * (1+grid_z)**2 + cospars.Omega_Lambda)
    if not hasattr(z, '__iter__'):
        DC = cospars.DH * sum(dz/E)
    else:
        # Hogg 2000 eqn 14,15
        sumE = [sum(dz/E[:n]) for n in xrange(0,len(E))]
        f = interp.interp1d(grid_z,sumE)
        DC = cospars.DH * f(z) 

    # flat universe
    DM = DC
    return DM

def get_ang_diam_dist(z1, z2=0.):
    """
    @brief Angular diameter distance, Hogg 2000, eq 19, z1 should be greater than z2, but if not, it get's it right anyway
    @return in Mpc
    """
    if (type(z1) is np.ndarray) and (type(z2) is np.ndarray):
        select1 = z1>=z2 
        select2 = z1<z2 
        new_z1 = z1 * select1 + z2 * select2
        new_z2 = z2 * select1 + z1 * select2
    else:
        new_z1 = z1
        new_z2 = z2        

    DM1 = get_comoving_dist_line_of_sight(new_z1) 
    DM2 = get_comoving_dist_line_of_sight(new_z2) 

    ang_diam_dist =  (DM1-DM2)/(1.+new_z1) 

    # if any(ang_diam_dist  < 0):
    #     import pdb; pdb.set_trace()

    return ang_diam_dist


def get_gnomonic_projection(ra_rad, de_rad , ra_center_rad, de_center_rad):
# http://mathworld.wolfram.com/GnomonicProjection.html

    cos_c = np.sin(de_center_rad)*np.sin(de_rad) + np.cos(de_center_rad) * np.cos(de_rad) * np.cos(ra_rad  - ra_center_rad)
    x = np.cos(de_rad)*np.sin(ra_rad - ra_center_rad)
    y = np.cos(de_center_rad) * np.sin(de_rad) - np.sin(de_center_rad)*np.cos(de_rad)*np.cos(ra_rad-ra_center_rad) 
    y = y/cos_c

    return x,y

def get_gnomonic_projection_shear(ra_rad, de_rad , ra_center_rad, de_center_rad, shear_g1 , shear_g2):

    # code from Jospeh
    # Here use a tangent plane to transform the sources' shear
    # from RA, DEC to a common local coordinate system around lens
    # phi_bar = ra_l[i] * pi/180.
    # th_bar = (90. - dec_l[i]) * pi/180.
    phi_bar = ra_center_rad
    th_bar = np.pi/2. - de_center_rad

    # phi = ra_s * (pi/180.)
    # theta = (90. - dec_s) * (pi/180.)
    phi = ra_rad
    theta = np.pi/2. - de_rad 

    gamma = np.abs(phi - phi_bar)

    cot_delta = ( np.sin(theta)  / np.tan(th_bar) - np.cos(theta)  * np.cos(gamma) ) /np.sin(gamma)
    cot_beta  = ( np.sin(th_bar) / np.tan(theta)  - np.cos(th_bar) * np.cos(gamma) ) /np.sin(gamma)
    delta = np.arctan(1./cot_delta)
    beta  = np.arctan(1./cot_beta )

    del_bar = np.pi - beta
    sgn = np.sign(phi - phi_bar)
    phase = np.exp(sgn *2.j *np.abs(delta - del_bar))

    # Now place shear in complex number, rotate by phase
    temp = (shear_g1 + 1.j*shear_g2) * phase

    gam1 = temp.real
    gam2 = temp.imag

    return gam1 , gam2





def get_angular_separation(ra1_rad,de1_rad,ra2_rad,de2_rad):

    d_ra =  np.abs(ra1_rad-ra2_rad)
    d_de =  np.abs(de1_rad-de2_rad)
    theta = np.arccos( np.sin(de1_rad)*np.sin(de2_rad) + np.cos(de1_rad)*np.cos(de2_rad)*np.cos(d_ra))
    # theta = 2*np.arcsin( np.sqrt( np.sin(d_de/2.)**2  + np.cos(de1_rad)*np.cos(de2_rad)*np.sin(d_ra/2.)**2 ) )


    return theta

def get_projection_matrix(center_ra,center_dec):

    delta = 1e-5

    ds_ra = get_angular_separation(0.,0.,delta,0.)
    ds_dec = get_angular_separation(0.,0.,0.,delta)

    P = [[ds_ra/delta, 0],[0,ds_dec/delta]]

    return P

    
def plot_DC_vs_z():

    z_vals=np.linspace(0,2,100);
    DC=[]
    for z in z_vals:
        DC.append(get_comoving_dist_line_of_sight(z))

    print cospars.DH
    import pylab as pl
    pl.plot(z_vals,DC)
    pl.show()

def spherical_to_cartesian_with_redshift(ra_deg,de_deg,z):

    if not hasattr(z, '__iter__'):
        xyz=np.zeros([1,3])        
    else:
        xyz=np.zeros([len(ra_deg),3])

    # import cosmolopy.distance as cd
    # cosmo = {'Omega_M_0' : 0.25, 'Omega_Lambda_0' : 0.75, 'h' : 0.72}
    # cosmo = cd.set_Omega_k_0(cosmo)
    # los = cd.comoving_distance_transverse(z, **cosmo) *  cosmo['h']
    # los = get_comoving_dist_line_of_sight(z)
    los = get_ang_diam_dist(z)
    ra_rad , de_rad = deg_to_rad(ra_deg, de_deg)
    x, y, z = spherical_to_cartesian_rad(ra_rad , de_rad , los)  

    return x, y, z

def spherical_to_cartesian_deg(ra_deg,de_deg,radius):

    ra_rad , de_rad = deg_to_rad(ra_deg, de_deg)
    return spherical_to_cartesian_rad(ra_rad,de_rad,radius)


def spherical_to_cartesian_rad(ra_rad,de_rad,radius):
 
    x=radius * np.cos(de_rad)*np.cos(ra_rad);
    y=radius * np.cos(de_rad)*np.sin(ra_rad);
    z=radius * np.sin(de_rad);
    return x, y, z

def cartesian_to_spherical_rad(x, y, z):

    r = np.sqrt(x**2 + y**2 + z**2)
    de_rad = np.pi/2. - np.arccos( z / r )
    ra_rad = np.arctan(y/x)   
    return ra_rad, de_rad, r

def cartesian_to_spherical_deg(x, y, z):    

    ra_rad , de_rad, r = cartesian_to_spherical_rad(x, y, z)
    ra_deg , de_deg =  rad_to_deg(ra_rad, de_rad) 
    return ra_deg , de_deg , r

def get_midpoint_deg( halo1_ra_deg , halo1_de_deg , halo2_ra_deg , halo2_de_deg ):

    halo1_ra_rad , halo1_de_rad = deg_to_rad(halo1_ra_deg , halo1_de_deg)
    halo2_ra_rad , halo2_de_rad = deg_to_rad(halo2_ra_deg , halo2_de_deg)
    pairs_ra_rad , pairs_de_rad = get_midpoint_rad( halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad )
    pairs_ra_deg , pairs_de_deg = rad_to_deg( pairs_ra_rad , pairs_de_rad )
    return pairs_ra_deg , pairs_de_deg


def get_midpoint_rad( halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad ):

    # eucl1_1, eucl2_1, eucl3_1 = spherical_to_cartesian_rad(halo1_ra_rad,halo1_de_rad,1)
    # eucl1_2, eucl2_2, eucl3_2 = spherical_to_cartesian_rad(halo2_ra_rad,halo2_de_rad,1)      
    # eucl1_mid = np.mean([eucl1_1,eucl1_2])
    # eucl2_mid = np.mean([eucl2_1,eucl2_2])
    # eucl3_mid = np.mean([eucl3_1,eucl3_2])
    # pairs_ra_rad , pairs_de_rad , _ = cartesian_to_spherical_rad(eucl1_mid,eucl2_mid,eucl3_mid)

    # lon <-> RA , lat <-> DEC
    Bx = np.cos(halo2_de_rad) * np.cos(halo2_ra_rad - halo1_ra_rad)
    By = np.cos(halo2_de_rad) * np.sin(halo2_ra_rad - halo1_ra_rad)
    pairs_de_rad = np.arctan2( np.sin(halo1_de_rad) + np.sin(halo2_de_rad) , np.sqrt( (np.cos(halo1_de_rad) + Bx)**2 + By**2 ) )
    pairs_ra_rad = halo1_ra_rad + np.arctan2(By , np.cos(halo1_de_rad) + Bx)

    # http://www.movable-type.co.uk/scripts/latlong.html

    return pairs_ra_rad , pairs_de_rad

def get_sigma_crit(z_gal,z_lens,unit='Msol*h/pc^2'):
    """
    @return Sigma_crit in specified units
    @param unit units to return Sigma_crit in, currently available 'kg/m^2' and 'Msol*h/pc^2'
    """

    DS = get_ang_diam_dist(z_gal)   # Mpc
    DL = get_ang_diam_dist(z_lens)  # Mpc
    DLS = get_ang_diam_dist(z_gal,z_lens)   # Mpc

    # Sigma_crit = cospars.c**2 / (4.*np.pi * cospars.G ) *  DS/(DL*DLS) / cospars.Mpc_to_m # kg/m^2
    Sigma_crit = cospars.c**2 / (4* np.pi * cospars.G) * DS / DLS / DL /cospars.M_solar * cospars.Mpc_to_m       

    if unit=='kg/m^2':
        return Sigma_crit
    elif unit=='Msol*h/pc^2':

        h = 1
        kg_per_Msol = 1.98892e30
        km_per_Mpc = 3.08568e19

        fac = km_per_Mpc / kg_per_Msol # kg/m --> Msol/kpc
        fac = fac * km_per_Mpc         # kg/m^2 --> Msol/kpc^2
        fac = fac * h / 1e6            # 1/kpc^2 --> h/pc^2

        return Sigma_crit * fac


def get_projection(vec_b,vec_a):

        
    a1 = np.dot(vec_a,vec_b.T) / np.dot(vec_b,vec_b.T) * vec_b
    a2 = vec_a - a1

    return a1,a2


    return u,v


def deg_to_rad(ra_deg,de_deg):

    ra_rad = ra_deg*np.pi/180.
    de_rad = de_deg*np.pi/180.    

    return ra_rad,de_rad

def rad_to_deg(ra_rad,de_rad):

    ra_deg = ra_rad * 180. / np.pi
    de_deg = de_rad * 180. / np.pi

    return ra_deg , de_deg


def arcsec_to_deg(ra_arcsec,de_arcsec):

    return ra_arcsec/3600. , de_arcsec/3600.

def deg_to_arcsec(ra_deg,de_deg):

    return ra_deg*3600, de_deg*3600

def deg_to_arcmin(ra_deg,de_deg):

    return ra_deg*60, de_deg*60


def rad_to_arcsec(ra_rad,de_rad):

    ra_arcsec = ra_rad/np.pi*180*3600. 
    de_arcsec = de_rad/np.pi*180*3600.

    return ra_arcsec , de_arcsec

def rad_to_arcmin(ra_rad,de_rad):

    ra_arcmin = ra_rad/np.pi*180*60. 
    de_arcmin = de_rad/np.pi*180*60.

    return ra_arcmin , de_arcmin

def rad_to_mpc(ra_rad,de_rad,z):

    ang_diam_dist = get_ang_diam_dist(z)
    ra_mpc = ra_rad*ang_diam_dist
    de_mpc = de_rad*ang_diam_dist

    return ra_mpc , de_mpc

def mpc_to_arcmin(ra_mpc,de_mpc,z):
    
    ang_diam_dist = get_ang_diam_dist(z)
    ra_rad = ra_mpc/ang_diam_dist
    de_rad = de_mpc/ang_diam_dist
    
    ra_arcmin , de_arcmin  = rad_to_arcmin(ra_rad,de_rad)

    return ra_arcmin, de_arcmin



if __name__=="__main__":

    plot_DC_vs_z()