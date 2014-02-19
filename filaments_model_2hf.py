class fitter():

    shear_g1 = None
    shear_g2 = None
    shear_u_arcmin = None
    shear_v_arcmin = None
    shear_z = 1000
    pairs_u1_arcmin = None
    pairs_v1_arcmin = None
    pairs_u2_arcmin = None
    pairs_v2_arcmin = None
    pairs_z1 = None
    pairs_z2 = None
    sigma_g = 0.2


    def draw_model(self,params):
        """
        params[0] mass log(M1)
        params[1] mass log(M2)
        params[2] filament_radius 
        params[3] filament_kappa 
        """

        # def draw_model(M1,M2,F,R,shears_info,pair_info):

        M1 = np.exp(params[0])
        M2 = np.exp(params[1])
        filament_radius = params[2]
        filament_kappa = params[3]

        c1 = self.get_concentr(M1,self.pairs_z1)
        c2 = self.get_concentr(M2,self.pairs_z2)
        halo1_pos = galsim.PositionD(x=self.pairs_u1_arcmin*60.,y=self.pairs_v1_arcmin*60.)
        halo2_pos = galsim.PositionD(x=self.pairs_u2_arcmin*60.,y=self.pairs_v2_arcmin*60.)
        shear_pos = ( self.shear_u_arcmin*60. , self.shear_v_arcmin*60.) 

        nfw1=galsim.NFWHalo(conc=c1, redshift=self.pairs_z1, mass=M1, omega_m = cosmology.cospars.omega_m, halo_pos=halo1_pos)
        nfw2=galsim.NFWHalo(conc=c2, redshift=self.pairs_z2, mass=M2, omega_m = cosmology.cospars.omega_m, halo_pos=halo2_pos)
        log.debug('getting lensing')
        h1g1 , h1g2 , _ =nfw1.getLensing(pos=shear_pos, z_s=self.shear_z)
        h2g1 , h2g2 , _ =nfw2.getLensing(pos=shear_pos, z_s=self.shear_z)
        fg1 , fg2 = self.filament_model(self.shear_u_arcmin,self.shear_v_arcmin,self.pairs_u1_arcmin,self.pairs_u2_arcmin,filament_kappa,filament_radius)

        model_g1 = h1g1 + h2g1 + fg1
        model_g2 = h1g2 + h2g2 + fg2
        return  model_g1 , model_g2 

    def lnpostfn(self,p):

        model_g1 , model_g2 = self.draw_model(p)
        chi2 = self.log_posterior(model_g1,model_g2)

    def log_posterior(self,model_g1,model_g2):

        chi2 = - np.sum( ((model_g1 - self.shear_g1)/self.sigma_g) **2) + np.sum( ((model_g2 - self.shear_g2)/self.sigma_g) **2)

        print chi2
        return chi2

    def run_mcmc(self):

        n_dim = 4
        n_walkers = 10

        log.info('getting sampler')
        sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=n_dim , lnpostfn=self.lnpostfn, a=2.0, args=[], threads=1, pool=None, live_dangerously=False)
        p0_walker= np.array([ 13.0, 14.0, 0.001, 0.05 ])
        p0 = [p0_walker for i in range(n_walkers)]
        sampler.run_mcmc(p0, 1000)

    def get_concentr(self,M,z):

        # Duffy et al 2008 from King and Mead 2011
        concentr = 5.72/(1.+z)**0.71 * (M / 1e14 * cospars.h)**(-0.081)
        return concentr

    def filament_model(self,shear_u_arcmin,shear_v_arcmin,u1_arcmin,u2_arcmin,kappa0,radius_arcmin):

        r = np.abs(shear_v_arcmin)

        kappa = - kappa0 / (1. + r / radius_arcmin)

        # zero the filament outside halos
        # we shoud zero it at R200, but I have to check how to calculate it from M200 and concentr
        select = ((shear_u_arcmin > u1_arcmin) + (shear_u_arcmin < u2_arcmin))
        kappa[select] *= 0.
        g1 = kappa
        g2 = g1*0.

        return g1 , g2


