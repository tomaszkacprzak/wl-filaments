import numpy as np
import pylab as pl
import sys,os, warnings, logging, cosmology, plotstools, tabletools

log = logging.getLogger("filament..") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False


cosmoparams = cosmology.cospars

class filament:

    def __init__(self):

        self.grid_z_edges = None
        self.grid_z_centers = None
        self.prob_z = None
        self.pair_z = None
        self.redshift_offset = 0.2
        self.mean_inv_sigma_crit = None
        self.scale_dens = 1e14
        self.shear_interp = None
        self.min_radius = 0.

    def proj_mass_density(self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,truncation=10):

        ## new ## ---------------------
        # amplitude = kappa0 /  (radius_mpc * np.pi) 

        # r = np.abs(shear_v_mpc)

        # dens = amplitude / (1. + (r / radius_mpc)**2 )

        # # zero the filament outside halos
        # # we shoud zero it at R200, but I have to check how to calculate it from M200 and concentr
        # select = ((shear_u_mpc > (u1_mpc) ) + (shear_u_mpc < (u2_mpc)  ))
        # dens[select] *= 0.

        # amplitude = kappa0 /  (radius_mpc * np.pi)  # use total mass 
        amplitude = kappa0

        truncation_radius = truncation * radius_mpc
        r = np.abs(shear_v_mpc)
        # dens = kappa0 / (1. + (r / radius_mpc)**2 )
        dens = (amplitude / (1. + (r/radius_mpc)**2) )  * np.cos(np.pi*r/truncation_radius/2)**2
        dens[r>truncation_radius]=0.
        dens *=  self.scale_dens

        # # zero the filament outside halos
        # # we shoud zero it at R200, but I have to check how to calculate it from M200 and concentr
        # select = ((shear_u_mpc > (u1_mpc) ) + (shear_u_mpc < (u2_mpc)  ))
        select = np.abs(shear_u_mpc) > np.abs(u1_mpc)
        dens[select] *= 0.


        return dens

    def get_filament_model_interp(self,kappa0,radius_mpc,pair_z, grid_z_centers,prob_z):

        radius_min = 0
        radius_max = 10
        v = np.linspace(radius_min,radius_max,1000)
        u = np.zeros_like(v)
       
        # (self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z, grid_z_centers , prob_z,  redshift_offset=0.2):
        g1 , g2 = self.filament_model_with_pz( u, v, -1, 1, kappa0, radius_mpc, pair_z, grid_z_centers, prob_z)
        pmdi = interp1d( v , g1 , 'nearest')

        return pmdi


    def filament_model(self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z,shear_z):
        """
        @brief create filament model
        @param shear_u_mpc array of u coordinates
        @param shear_v_mpc array of v coordinates
        @param u1_mpc start of the filament 
        @param u2_mpc end of the filament 
        @param kappa0 total mass of the filament 
        """

        dens = self.proj_mass_density(shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc)
        sigma_crit = cosmology.get_sigma_crit(shear_z,pair_z,unit='kg/m^2')
        # print 'sigma_crit' , sigma_crit
        # print 'max dens' , max(dens)
        kappa = dens / sigma_crit 
        g1 = -kappa
        g2 = np.zeros_like(kappa)

        return g1 , g2

    def get_shear_profile_interp(self,radius_mpc,theta):

        if self.shear_interp == None:
            log.info('loading shear profiles')
            filename_profiles = 'shear_profiles.pp2'
            profiles_dict = tabletools.loadPickle(filename_profiles)
            import scipy.interpolate
            self.shear_interp = scipy.interpolate.interp2d(y=profiles_dict['grid_radius'],x=profiles_dict['grid_theta'],z=profiles_dict['profiles'])

        # import pdb; pdb.set_trace()
        # pl.plot(profiles_dict['grid_theta'],self.shear_interp(3,profiles_dict['grid_theta'])); pl.show()
        profile_g = self.shear_interp(theta,radius_mpc) 

        return profile_g

    def get_shear_profile(self,radius,max_radius,n_grid=2000):

        # na = len(shear_u_mpc)
        # nv = 80
        # nu = na/nv

        x=np.linspace(-max_radius,max_radius,n_grid)
        y=np.linspace(-max_radius,max_radius,n_grid)
        pixel_area = (x[1] - x[0])**2
        X,Y = np.meshgrid(x,y,indexing='ij')
 
        dens = self.proj_mass_density(X,Y,max_radius,-max_radius,1.,radius , truncation=10)

        # pl.figure()
        # pl.plot(Y[0,:],dens[0,:])
        # pl.yscale('log')
        # pl.show()
        
        D =D_kernel = -1. / (X  - 1j*Y)**2

        # pl.figure()
        # pl.pcolormesh(x,y,D.real)
        # pl.colorbar()

        # pl.figure()
        # pl.pcolormesh(x,y,D.imag)
        # pl.colorbar()

        # pl.figure()
        # pl.pcolormesh(x,y,dens)
        # pl.colorbar()

        import scipy.signal
        print 'convolving'
        print D.shape, dens.shape

        sumD1 = np.sum(D,axis=0)
        yy = Y[0,:]
        kk = dens[0,:]
        sumD1_truncation = 40
        # sumD1 = sumD1*np.cos(np.pi*yy/sumD1_truncation/2.)**2
        sumD1[np.abs(yy)>sumD1_truncation]= 0.

        # pl.figure()
        # pl.plot(yy,kk)

        # pl.figure()
        # pl.plot(yy,np.abs(sumD1))
        # # pl.xscale('log')
        # pl.yscale('log')
        # pl.show()

        conv = scipy.signal.convolve(kk,sumD1,mode='same') * pixel_area /np.pi

        # pl.figure()
        # pl.plot(yy,conv.real,'b.-')
        # pl.plot(yy,conv.imag,'r.-')
        # pl.xscale('log')
        # pl.yscale('log')
        # pl.show()
        
        return conv, yy, kk


    def set_mean_inv_sigma_crit(self,grid_z_centers,prob_z,pair_z):

        sigma_crit = cosmology.get_sigma_crit(grid_z_centers,pair_z,unit='kg/m^2')
        prob_z_limit = prob_z
        prob_z_limit[grid_z_centers < pair_z + self.redshift_offset] = 0
        prob_z_limit /= sum(prob_z_limit)
        self.mean_inv_sigma_crit = sum(prob_z_limit / sigma_crit)


    def filament_model_with_pz(self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z, grid_z_centers , prob_z,  redshift_offset=0.2):


        abs_kappa0 = np.abs(kappa0)
        abs_radius = np.abs(radius_mpc) + self.min_radius
        dens = self.proj_mass_density(shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,abs_kappa0,abs_radius)
        kappa = dens * self.mean_inv_sigma_crit * np.sign(kappa0)
        g1 = -kappa
        g2 = np.zeros_like(kappa)

        return g1 , g2

    def filament_model_with_pz_new(self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z, grid_z_centers , prob_z, redshift_offset=0.2 ):


        profile_g = self.get_shear_profile_interp(radius_mpc,shear_v_mpc)
        g1 = profile_g * self.mean_inv_sigma_crit * kappa0
        g2 = np.zeros_like(g1)

        # zero profile outside limits
        select = ((shear_u_mpc > (u1_mpc) ) + (shear_u_mpc < (u2_mpc)  ))
        g1[select] *= 0.
        
        return g1 , g2

    def get_bcc_pz(self,filename_lenscat):

        if self.prob_z == None:

            # filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'
            # filename_lenscat = os.environ['HOME'] + '/data/BCC/bcc_a1.0b/aardvark_v1.0/lenscats/s2n10cats/aardvarkv1.0_des_lenscat_s2n10.351.fit'

            if 'fits' in filename_lenscat:
                lenscat = tabletools.loadTable(filename_lenscat)
                if 'z' in lenscat.dtype.names:
                    self.prob_z , _  = pl.histogram(lenscat['z'],bins=self.grid_z_edges,normed=True)
                elif 'z-phot' in lenscat.dtype.names:
                    self.prob_z , _  = pl.histogram(lenscat['z-phot'],bins=self.grid_z_edges,normed=True)

                if 'e1' in lenscat.dtype.names:

                    select = lenscat['star_flag'] == 0
                    lenscat = lenscat[select]
                    select = lenscat['fitclass'] == 0
                    lenscat = lenscat[select]
                    select = (lenscat['e1'] != 0.0) * (lenscat['e2'] != 0.0)
                    lenscat = lenscat[select]
                    self.sigma_ell = np.std(lenscat['e1']*lenscat['weight'],ddof=1)

            elif 'pp2' in filename_lenscat:

                pickle = tabletools.loadPickle(filename_lenscat)
                self.prob_z =  pickle['prob_z']
                self.grid_z_centers = pickle['bins_z']
                self.grid_z_edges = plotstools.get_bins_edges(self.grid_z_centers)
                self.sigma_ell = pickle['sigma_ell']
                log.info('loaded sigma_ell=%2.2f', self.sigma_ell)



def test():

    # mass = np.logspace(0,15,100)
    mass = np.linspace(0, 4e2 ,10)
    n_points = 100
    shear_u_mpc = np.ones(n_points)*0
    shear_v_mpc = np.linspace(-10,10,n_points)
    u1_mpc = 10
    u2_mpc = -10
    kappa0 = None
    radius_mpc = 1

    f = filament()
    f.pair_z = 0.6
    f.n_points = len(shear_u_mpc)
    f.redshift_offset = 0.2

    f.get_bcc_pz()
    f.set_mean_inv_sigma_crit(f.grid_z_centers,f.prob_z,f.pair_z)


    kappa0 = 50
    radius_mpc = 0.5
    model_pz_1 , model_pz_2 = f.filament_model_with_pz_new( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'ro')
    print min(model_pz_1)

    kappa0 = 150
    radius_mpc = 3
    model_pz_1 , model_pz_2 = f.filament_model_with_pz_new( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'bx',ms=20)
    print min(model_pz_1)

    kappa0 = 150
    radius_mpc = 0.5
    model_pz_1 , model_pz_2 = f.filament_model_with_pz_new( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'gd')
    print min(model_pz_1)

    kappa0 = 50
    radius_mpc = 3
    model_pz_1 , model_pz_2 = f.filament_model_with_pz_new( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'m+',ms=20)
    print min(model_pz_1)

    # pl.yscale('log')
    pl.show()

def test_shear_profile():

    nu = 100
    nv = 80
    shear_u_mpc = np.linspace(-10,10,nu)
    shear_v_mpc = np.linspace(-4,4,nv)

    U,V = np.meshgrid(shear_u_mpc,shear_v_mpc)
    shear_u_mpc,shear_v_mpc = U.flatten() , V.flatten()

    u1_mpc = 8
    u2_mpc = -8
    kappa0 = None
    radius_mpc = 1

    f = filament()
    f.pair_z = 0.6
    f.n_points = len(shear_u_mpc)
    f.redshift_offset = 0.2

    filename_lenscat = 'CFHTLens_2014-06-14.normalised.pz.pp2'
    f.get_bcc_pz(filename_lenscat)
    f.set_mean_inv_sigma_crit(f.grid_z_centers,f.prob_z,f.pair_z)


    kappa0 = 50
    radius_mpc = 1
    model_pz_1 , model_pz_2 = f.filament_model_with_pz( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    # def filament_model_with_pz(self,shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc,pair_z, grid_z_centers , prob_z,  redshift_offset=0.2):    pl.sublot(1,2,1)

    pl.figure()
    pl.subplot(1,2,1)
    pl.scatter(shear_u_mpc , shear_v_mpc, c=model_pz_1 )
    pl.subplot(1,2,2)
    pl.scatter(shear_u_mpc , shear_v_mpc, c=model_pz_2 )
    print min(model_pz_1)

    # pl.show()

    pl.figure()

    kappa0 = 0.05
    radius_mpc = 1
    model_pz_1 , model_pz_2 = f.filament_model_with_pz( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'bx',ms=10)
    print min(model_pz_1)

    kappa0 = 0.05
    radius_mpc = 3
    model_pz_1 , model_pz_2 = f.filament_model_with_pz( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'gd',ms=10)
    print min(model_pz_1)

    kappa0 = 0.05
    radius_mpc = 5
    model_pz_1 , model_pz_2 = f.filament_model_with_pz( shear_u_mpc,shear_v_mpc,u1_mpc,u2_mpc,kappa0,radius_mpc, f.pair_z, f.grid_z_centers , f.prob_z)
    pl.plot(shear_v_mpc,-model_pz_1,'m+',ms=10)
    print min(model_pz_1)

    # pl.yscale('log')
    pl.show()


def get_shear_lookup():

    f = filament()
    n_grid=5000
    max_radius = 30
    grid_radius = np.linspace(0.001,max_radius,100)
    x=np.linspace(-max_radius,max_radius,n_grid)
    y=np.linspace(-max_radius,max_radius,n_grid)  
    pixel_area = (x[1] - x[0])**2
    X,Y = np.meshgrid(x,y,indexing='ij')
    D =D_kernel = -1. / (X  - 1j*Y)**2
    sumD1 = np.sum(D,axis=0)
    import scipy.signal

    list_profiles = []
    for rad in grid_radius:
        print 'rad=' , rad
        kappa0 = 1.
        dens = f.proj_mass_density(X,Y,max_radius,-max_radius,kappa0,rad)
        conv = scipy.signal.convolve(dens[0,:],sumD1,mode='same') * pixel_area
        list_profiles.append(conv)

    profiles = np.array(list_profiles)

    save_dict = { 'profiles' : profiles , 'grid_radius' : grid_radius , 'grid_theta' :  y}
    
    filename_profiles = 'shear_profiles.pp2'
    tabletools.savePickle(filename_profiles,save_dict)

    pl.figure()
    pl.imshow(profiles.real,aspect='auto',interpolation='nearest')
    pl.colorbar()

    pl.figure()
    pl.imshow(profiles.imag,aspect='auto',interpolation='nearest')
    pl.colorbar()
    pl.show()


def get_shear_profiles_plot():

    f = filament()
    radius_max = 300
    shear_z = 1
    pair_z = 0.2
    kappa = 1.
    n_grid = 3000

    ii=0
    for radius_max in [50]:

        n_grid = radius_max*100

        for radius in [1,2,5,10]:
            print 'radius' , radius , 'radius_max' , radius_max
            shear_part, grid_radius, dens = f.get_shear_profile(radius,radius_max,n_grid)
            # sigma_crit = cosmology.get_sigma_crit(shear_z,pair_z,unit='kg/m^2')   
            # shear = shear_part / sigma_crit * kappa
            shear = shear_part * kappa / f.scale_dens
            g1 = shear.real 
            # pl.plot(grid_radius,g1,'b-')

            # g1 = f.get_shear_profile_interp(radius,grid_radius) / sigma_crit * kappa
            # pl.plot(grid_radius,g1,'r-')
            if ii==0:
                pl.plot(grid_radius,kappa/(1.+(grid_radius/radius)**2),label='kappa')
                ii=1
            pl.plot(grid_radius,np.abs(g1),label='max_radius=%2.2f, radius=%2.2f' % (radius_max,radius))
            print max(np.abs(g1))


    pl.xlim(0.1,radius_max)
    pl.ylim(0.01,10)
    pl.grid()
    pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.xlabel('radius')
    pl.ylabel('shear')
    pl.title('radius_core=%0.2f kappa0=%.2f' % (radius,kappa))
    pl.show()

def get_2d_fiament_shear():

    radius_max = 20
    max_radius = 20
    shear_z = 1
    pair_z = 0.2
    kappa0 = 0.01
    n_grid = 100

    x=np.linspace(-max_radius,max_radius,n_grid)
    y=np.linspace(-max_radius,max_radius,n_grid)
    pixel_area = (x[1] - x[0])**2
    X,Y = np.meshgrid(x,y,indexing='ij')

    # import nfw
    # n1 = nfw.NfwHalo()
    # n1.theta_cx = 9               # center of the halo in arcmin
    # n1.theta_cy = 0
    # gamma_1 , gamma_2 , Delta_Sigma_1, Delta_Sigma_2 , Sigma_crit , kappa =n1.get_shears(X,Y,1.)   
    # pl.figure()
    # from matplotlib.colors import LogNorm
    # pl.pcolormesh(X,Y,kappa,norm = LogNorm())
    # pl.colorbar()

    radius=2.
    f = filament()
    dens = f.proj_mass_density(X,Y,max_radius*0.8,-max_radius*0.8,kappa0,radius) / f.scale_dens
    D =D_kernel = -1. / (X  - 1j*Y)**2

    pl.figure()
    pl.pcolormesh(x,y,D.real)
    pl.colorbar()

    pl.figure()
    pl.pcolormesh(x,y,D.imag)
    pl.colorbar()

    pl.figure()
    pl.pcolormesh(X,Y,dens)
    pl.colorbar()

    import scipy.signal
    print 'convolving'
    print D.shape, dens.shape

    conv = scipy.signal.convolve2d(D,dens,mode='same') * pixel_area /np.pi
    g1=conv.real
    print g1[:].min()
    pl.figure()
    pl.pcolormesh(X,Y,g1)
    pl.colorbar()
    pl.show()

    import pdb; pdb.set_trace()



if __name__=='__main__':

    # get_shear_lookup()
    # get_shear_profiles_plot()
    # test()
    test_shear_profile()
    # get_2d_fiament_shear()
