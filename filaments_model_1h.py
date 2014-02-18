import yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools
import numpy as np
import pylab as pl


log = logging.getLogger("fil_mod_1h") 
log.setLevel(logging.DEBUG)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

class modelfit():

    def __init__(self):
        
        self.shear_g1 = None
        self.shear_g2 = None
        self.shear_u_arcmin = None
        self.shear_v_arcmin = None
        self.shear_z = 10
        self.halo_u_arcmin = None
        self.halo_v_arcmin = None
        self.halo_z = None
        self.sigma_g = 0.2
        self.n_model_evals = 0
        self.sampler = None

    def plot_shears(self,g1,g2):
        
        emag=np.sqrt(g1**2+g2**2)
        ephi=0.5*np.arctan2(g2,g1)              

        nuse=1
        quiver_scale=20
        pl.quiver(self.shear_u_arcmin[::nuse],self.shear_v_arcmin[::nuse],emag[::nuse]*np.cos(ephi)[::nuse],emag[::nuse]*np.sin(ephi)[::nuse],linewidths=0.005,headwidth=0., headlength=0., headaxislength=0., pivot='mid',color='r',label='original',scale=quiver_scale)
        
        pl.xlim([min(self.shear_u_arcmin),max(self.shear_u_arcmin)])
        pl.ylim([min(self.shear_v_arcmin),max(self.shear_v_arcmin)])
        pl.axis('equal')

    def plot_residual(self,model_g1,model_g2):

        pl.figure()
        pl.subplot(3,1,1)
        self.plot_shears(self.shear_g1,self.shear_g2)
        pl.subplot(3,1,2)
        self.plot_shears(model_g1,model_g2)
        pl.subplot(3,1,3)
        self.plot_shears(model_g1-self.shear_g1,model_g2-self.shear_g2)
        pl.show()


    def draw_model(self,params):
        """
        params[0] mass log(M)
        """

        self.n_model_evals +=1

        M200 = 10.**params[0]
        # conc = params[1]
        conc = self.get_concentr(M200,self.halo_z)

        import pdb; pdb.set_trace()

        halo_pos = galsim.PositionD(x=self.halo_u_arcmin*60.,y=self.halo_v_arcmin*60.)
        shear_pos = ( self.shear_u_arcmin*60. , self.shear_v_arcmin*60.) 

        nfw=galsim.NFWHalo(conc=conc, redshift=self.halo_z, mass=M200, omega_m = cosmology.cospars.omega_m, halo_pos=halo_pos)
        # log.debug('getting lensing for single NFW')
        h1g1 , h1g2 , _ =nfw.getLensing(pos=shear_pos, z_s=self.shear_z)

        model_g1 = h1g1 
        model_g2 = h1g2 

        self.plot_residual(model_g1,model_g2)

        return  model_g1 , model_g2 

    def lnpostfn(self,p):

        model_g1 , model_g2 = self.draw_model(p)
        chi2 = self.log_posterior(model_g1,model_g2)
        # chi2 = -0.5 * np.sum((p[0] - 3) ** 2)


        log.debug('%7d chi2=% 2.4e % 2.10e' % (self.n_model_evals,chi2,p[0]))
        return chi2

    def log_posterior(self,model_g1,model_g2):

        chi2 = - ( np.sum( ((model_g1 - self.shear_g1)/self.sigma_g) **2) - np.sum( ((model_g2 - self.shear_g2)/self.sigma_g) **2) )

        return chi2

    def run_mcmc(self,):

        n_dim = 1 
        n_walkers = 100
        self.n_model_evals = 0

        log.info('getting self.sampler')
        self.sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=n_dim , lnpostfn=self.lnpostfn)
        p0_walker= np.array([ 14.0  ]) 
        p0 = [p0_walker+np.random.uniform(2) for i in range(n_walkers)]
        print p0

        self.sampler.run_mcmc(p0, 100)

    def get_concentr(self,M,z):

        # Duffy et al 2008 from King and Mead 2011
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cosmology.cospars.h)**(-0.081)
        return concentr
