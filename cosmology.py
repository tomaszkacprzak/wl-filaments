import logging
import numpy as np
import scipy.interpolate as interp

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('cosmology')
logger.setLevel(logging.INFO)

class cosmoparams:

    def __init__(self):

        self.omega_m = 0.3
        self.omega_lambda = 0.7
        self.w = -1.
        self.omega_k = 0.
        self.h = 0.7
        self.H0 = 100.*self.h # 
        self.DH= 3000.       # Mpc
        self.G = 6.673e-11; # SI units m3 kg-1 s-2
        self.c =  299792458; # m/s
        self.mpc_to_m = 3.0856e22; # number of meters in 1Mpc
        self.m_solar = 1.989e30; # kg / M_solar

        # G = 6.672e-8;
        # C = 29979245800;
        # Pc = 3.0857e18;

cospars = cosmoparams()


def get_comoving_dist_line_of_sight(z):
    """
    @return in Mpc
    """

    n_grid_z = 1000
    try:
        maxz = max(z)
    except TypeError, te:
        maxz = z
        
    grid_z = np.linspace(0,maxz,n_grid_z)
    dz = grid_z[1]
    E = np.sqrt(cospars.omega_m * (1+grid_z)**3 + cospars.omega_k * (1+grid_z)**2 + cospars.omega_lambda)
    if not hasattr(z, '__iter__'):
        DC = cospars.DH * sum(dz/E)
    else:
        # Hogg 2000 eqn 14,15
        sumE = [sum(dz/E[:n]) for n in xrange(0,len(E))]
        f = interp.interp1d(grid_z,sumE)
        DC = cospars.DH * f(z) * cospars.h

    # flat universe
    DM = DC
    return DM

def get_ang_diam_dist(z1, z2=0.):
    """
    @brief Angular diameter distance, Hogg 2000, eq 19, z1 should be greater than z2, but if not, it get's it right anyway
    @return in Mpc
    """
    if (type(z1) is np.ndarray) and (type(z2) is np.ndarray):
        select1 = z1>z2 
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



def get_angular_separation(ra1,dec1,ra2,dec2):

    theta = np.arccos( np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2))
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

def get_euclidian_coords(ra_deg,de_deg,z):

    ra_rad = ra_deg*np.pi/180.
    de_rad = (90.-de_deg)*np.pi/180.

    if not hasattr(z, '__iter__'):
        xyz=np.zeros([1,3])        
    else:
        xyz=np.zeros([len(ra_deg),3])

    # import cosmolopy.distance as cd
    # cosmo = {'omega_M_0' : 0.25, 'omega_lambda_0' : 0.75, 'h' : 0.72}
    # cosmo = cd.set_omega_k_0(cosmo)
    # los = cd.comoving_distance_transverse(z, **cosmo) *  cosmo['h']
    # los = get_comoving_dist_line_of_sight(z)
    los = get_ang_diam_dist(z)
    xyz[:,0]=los*np.sin(de_rad)*np.cos(ra_rad);
    xyz[:,1]=los*np.sin(de_rad)*np.sin(ra_rad);
    xyz[:,2]=los*np.cos(de_rad);

    return xyz

def euclidian_to_radec(x,y,z):

    r = np.sqrt(x**2 + y**2 + z**2)
    de = 90 - np.arccos( z / r )
    ra = arctan(y/x) 

    return ra,de,r

def get_sigma_crit(z_gal,z_lens,unit='kg/m^2'):
    """
    @return Sigma_crit in specified units
    @param unit units to return Sigma_crit in, currently available 'kg/m^2' and 'Msol*h/pc^2'
    """

    DS = get_ang_diam_dist(z_gal)   # Mpc
    DL = get_ang_diam_dist(z_lens)  # Mpc
    DLS = get_ang_diam_dist(z_gal,z_lens)   # Mpc

    Sigma_crit = cospars.c**2 / (4.*np.pi * cospars.G ) *  DS/(DL*DLS) / cospars.mpc_to_m # kg/m^2

    if unit=='kg/m^2':
        return Sigma_crit
    elif unit=='Msol*h/pc^2':

        h = 1
        kg_per_Msol = 1.98892e30
        km_per_mpc = 3.08568e19

        fac = km_per_mpc / kg_per_Msol # kg/m --> Msol/kpc
        fac = fac * km_per_mpc         # kg/m^2 --> Msol/kpc^2
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






if __name__=="__main__":

    plot_DC_vs_z()