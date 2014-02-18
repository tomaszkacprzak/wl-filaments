import pyfits
import numpy as np
import pylab as pl
import scipy.interpolate as interp
import cosmology
import tabletools
import yaml, argparse, sys, logging 
from sklearn.neighbors import BallTree as BallTree
import galsim
import filaments_tools
import emcee


logging.basicConfig(filename='filament_fit.log',level=logging.DEBUG,format='%(message)s')
logger = logging.getLogger("get_halo_pairs") 
logger.setLevel(logging.DEBUG)


cospars = cosmology.cosmoparams()

def mass_concentr_relation():

    pass

def grid_search(pair_info,shears_info):

    n_grid = 10
    range_M = np.linspace(1.9e14 , 1e15 , n_grid)
    range_F = np.linspace(0.01 , 0.1 , n_grid)
    range_R = np.linspace(0.5 , 2, n_grid)

    z = pair_info['z']
    data_g1 = shears_info['g1']
    data_g2 = shears_info['g2']
    

    for iM1,vM1 in enumerate(range_M):
        for iM2,vM2 in enumerate(range_M):
            for iF,vF in enumerate(range_F):
                for iR,vR in enumerate(range_R):

                    model_g1 , model_g2 = draw_model(vM1,vM2,vF,vR,shears_info,pair_info)

                    chi2 = likelihood(model_g1,model_g2,data_g1,data_g2,sigma_g = 0.01)

                    logger.debug('chi2=%2.2f M1=%1.4e M2=%1.4e F=%1.4e R=%1.4e ' % (chi2,vM1,vM2,vF,vR))

                    # filaments_tools.plot_pair(pair_info['u1_arcmin'],pair_info['v1_arcmin'],pair_info['u2_arcmin'],pair_info['v2_arcmin'],shears_info['u_arcmin'],shears_info['v_arcmin'],model_g1,model_g2,idp=0,nuse=10,tag='test',show=True,close=True)



    

def draw_model(M1,M2,F,R,shears_info,pair_info):

    z = pair_info['z']
    c1 = get_concentr(M1,z)
    c2 = get_concentr(M2,z)
    halo1_pos = galsim.PositionD(x=pair_info['u1_arcmin']*60.,y=pair_info['v1_arcmin']*60.)
    halo2_pos = galsim.PositionD(x=pair_info['u2_arcmin']*60.,y=pair_info['v2_arcmin']*60.)
    shear_pos = ( shears_info['u_arcmin']*60. , shears_info['v_arcmin']*60.) 

    nfw1=galsim.NFWHalo(conc=c1,redshift=z,mass=M1,omega_m = cosmology.cospars.omega_m,halo_pos=halo1_pos)
    nfw2=galsim.NFWHalo(conc=c2,redshift=z,mass=M2,omega_m = cosmology.cospars.omega_m,halo_pos=halo2_pos)
    logger.debug('getting lensing')
    h1g1 , h1g2 , _ =nfw1.getLensing(pos=shear_pos,z_s=shears_info['z'])
    h2g1 , h2g2 , _ =nfw2.getLensing(pos=shear_pos,z_s=shears_info['z'])
    fg1 , fg2 = filament_model(shears_info['u_arcmin'],shears_info['v_arcmin'],shears_info['scinv'],pair_info['u1_arcmin'],pair_info['u2_arcmin'],F,R)

    model_g1 = h1g1 + h2g1 + fg1
    model_g2 = h1g2 + h2g2 + fg2
    return  model_g1 , model_g2 


def filament_model(shear_u_arcmin,shear_v_arcmin,u1_arcmin,u2_arcmin,kappa0,radius_arcmin):

    r = np.abs(shear_v_arcmin)

    kappa = - kappa0 / (1. + r / radius_arcmin)

    # zero the filament outside halos
    # we shoud zero it at R200, but I have to check how to calculate it from M200 and concentr
    select = ((shear_u_arcmin > u1_arcmin) + (shear_u_arcmin < u2_arcmin))
    kappa[select] *= 0.
    g1 = kappa
    g2 = g1*0.

    return g1 , g2

def test_model(shears_info,pair_info):

    # def plot_pair(halo1_ra,halo1_de,halo2_ra,halo2_de,shear_ra,shear_de,shear_g1,shear_g2,idp=0,nuse=10,tag='test'):

    
    M1 , M2 = 2e14 , 2e14
    kappa0 = 0.05
    radius = 1
    model_g1 , model_g2 = draw_model(M1,M2,kappa0,radius,shears_info,pair_info)

    filaments_tools.plot_pair(pair_info['u1_arcmin'],pair_info['v1_arcmin'],pair_info['u2_arcmin'],pair_info['v2_arcmin'],shears_info['u_arcmin'],shears_info['v_arcmin'],model_g1,model_g2,idp=0,nuse=10,tag='test',show=False,close=False)
    filaments_tools.plot_pair(pair_info['u1_arcmin'],pair_info['v1_arcmin'],pair_info['u2_arcmin'],pair_info['v2_arcmin'],shears_info['u_arcmin'],shears_info['v_arcmin'],shears_info['g1'],shears_info['g2'],idp=0,nuse=10,tag='test',show=False,close=False)

    pl.show()














class fitter():

    shear_g1 = None
    shear_g2 = None
    shear_u_arcmin = None
    shear_v_arcmin = None
    shear_z = None
    pairs_u1_arcmin = None
    pairs_v1_arcmin = None
    pairs_u2_arcmin = None
    pairs_v2_arcmin = None
    pairs_z1 = None
    pairs_z2 = None
    sigma_g = 0.2
    data_g1 = None
    data_g2 = None

    def draw_model(params):

        # def draw_model(M1,M2,F,R,shears_info,pair_info):

        M1 = params[0]
        M2 = params[1]
        filament_radius = params[2]
        filament_kappa = params[3]

        c1 = self.get_concentr(M1,self.pairs_z1)
        c2 = self.get_concentr(M2,self.pairs_z2)
        halo1_pos = galsim.PositionD(x=self.pairs_u1_arcmin*60.,y=self.pairs_v1_arcmin*60.)
        halo2_pos = galsim.PositionD(x=self.pairs_u2_arcmin*60.,y=self.pairs_v2_arcmin*60.)
        shear_pos = ( self.shears_u_arcmin*60. , self.shears_v_arcmin*60.) 

        nfw1=galsim.NFWHalo(conc=c1, redshift=self.pairs_z1, mass=M1, omega_m = cosmology.cospars.omega_m, halo_pos=halo1_pos)
        nfw2=galsim.NFWHalo(conc=c2, redshift=self.pairs_z2, mass=M2, omega_m = cosmology.cospars.omega_m, halo_pos=halo2_pos)
        logger.debug('getting lensing')
        h1g1 , h1g2 , _ =nfw1.getLensing(pos=shear_pos, z_s=self.shears_z)
        h2g1 , h2g2 , _ =nfw2.getLensing(pos=shear_pos, z_s=self.shears_z)
        fg1 , fg2 = filament_model(self.shears_u_arcmin,self.shears_v_arcmin,self.pairs_u1_arcmin,,self.pairs_u2_arcmin,filament_kappa,filament_radius)

        model_g1 = h1g1 + h2g1 + fg1
        model_g2 = h1g2 + h2g2 + fg2
        return  model_g1 , model_g2 

    def lnpostfun(p, args):

        model_g1 , model_g2 = draw_model(p)
        chi2 = log_posterior(model_g1,model_g2)

    def log_posterior(model_g1,model_g2):

        chi2 = np.sum( ((model_g1 - self.data_g1)/self.sigma_g) **2) + np.sum( ((model_g2 - self.data_g2)/self.sigma_g) **2)

        return chi2

    def run_mcmc():

        n_dim = 5
        n_walkers = 10

        sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=n_dim , lnpostfun=self.lnpostfun, a=2.0, args=[], threads=1, pool=None, live_dangerously=False)

    def get_concentr(M,z):

        # Duffy et al 2008 from King and Mead 2011
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cospars.h)**(-0.081)
        return concentr








def main():

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    # parser.add_argument('-c', '--filename_config', default='test2.yaml',type=str, action='store', help='name of the yaml config file')
    # parser.add_argument('-d', '--dry', default=False,  action='store_true', help='Dry run, dont generate data')

    args = parser.parse_args()
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    logger = logging.getLogger("filaments_fit") 
    logger.setLevel(logging_level)

    filename_shears = 'shears_bcc_g.fits'
    filename_pairs = 'pairs_bcc.fits'

    pairs_table = tabletools.loadTable(filename_pairs)
    shears_hdus = pyfits.open(filename_shears)

    id_pair = 52
    shears_info = np.array(shears_hdus[id_pair+1].data)

    import pdb;pdb.set_trace()

    fitobj = fitter()
    fitobj.shear_g1 =  shears_info['g1']
    fitobj.shear_g2 =  shears_info['g2']
    fitobj.shear_u_arcmin =  shears_info['u_arcmin']
    fitobj.shear_v_arcmin =  shears_info['v_arcmin']
    fitobj.shear_z =  shears_info['shear_z']
    fitobj.pairs_u1_arcmin =  shears_info['u1_arcmin']
    fitobj.pairs_v1_arcmin =  shears_info['v1_arcmin']
    fitobj.pairs_u2_arcmin =  shears_info['u2_arcmin']
    fitobj.pairs_v2_arcmin =  shears_info['v2_arcmin']
    fitobj.pairs_z1 =  pairs_info['z1']
    fitobj.pairs_z2 =  pairs_info['z2']
    fitobj.sigma_g =  0.2
    fitobj.data_g1 =  shears_info['data_g1']
    fitobj.data_g2 =  shears_info['data_g2']


    pair_info = pairs_table[id_pair]



    # grid_search(pair_info,shears_info)
    # test_model(shears_info,pair_info)


main()