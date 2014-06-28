import os
if os.environ['USER'] == 'ucabtok': os.environ['MPLCONFIGDIR']='.'
import matplotlib as mpl
if 'DISPLAY' not in os.environ:
    mpl.use('agg')
import os, yaml, argparse, sys, logging , pyfits, tabletools, cosmology, filaments_tools, plotstools, mathstools, scipy, scipy.stats, time
import numpy as np
import pylab as pl
print 'using matplotlib backend' , pl.get_backend()
# import matplotlib as mpl;
# from matplotlib import figure;
pl.rcParams['image.interpolation'] = 'nearest' ; 
import scipy.interpolate as interp
from sklearn.neighbors import BallTree as BallTree
import cPickle as pickle
import filaments_model_1h
import filaments_model_1f
import filaments_model_2hf
import shutil
try:
    from pyqt_fit import kde
except: 
    print 'importing pyqt_fit failed'

log = logging.getLogger("filam..fit") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

cospars = cosmology.cosmoparams()

def get_stamps():
    
    filename_halos = config['filename_allhalos_cat']
    filename_shear = config['filename_allhalos_shears']
    halos = tabletools.loadTable(filename_halos)
    halos = tabletools.ensureColumn(rec=halos,arr=range(len(halos)),name='index',dtype='i4')
    log.info('halos %d' , len(halos))

    id_first = args.first
    id_last = args.first + args.num
   
    box_size=30 # arcmin
    pixel_size=0.5
    vec_u_arcmin, vec_v_arcmin = np.arange(-box_size/2.,box_size/2.+1e-9,pixel_size), np.arange(-box_size/2.,box_size/2.+1e-9,pixel_size)
    grid_u_arcmin, grid_v_arcmin = np.meshgrid( vec_u_arcmin  , vec_v_arcmin ,indexing='ij')
    vec_u_rad , vec_v_rad   = cosmology.arcmin_to_rad(vec_u_arcmin,vec_v_arcmin)
    grid_u_rad , grid_v_rad = cosmology.arcmin_to_rad(grid_u_arcmin,grid_v_arcmin)
    binned_shear_g1 = np.zeros_like(grid_u_arcmin)
    binned_shear_g2 = np.zeros_like(grid_u_arcmin) 
    binned_shear_n  = np.zeros_like(grid_u_arcmin) 
    binned_shear_w  = np.zeros_like(grid_u_arcmin) 
    binned_shear_m  = np.zeros_like(grid_u_arcmin) 

    cfhtlens_shear_catalog=None
    for ih in range(id_first, id_last):


        ihalo = halos['index'][ih]
        vhalo = halos[ih]

        filename_results_prob = 'results.prob.%04d.%04d.fits' % (id_first, id_last) 
        if not os.path.isfile(filename_results_prob):
            halos_results = np.empty(args.num,dtype={'names':['ihalo','m200_fit','log_prob'],'formats':['i4','f4','%df4'%config['M200']['n_grid']]})
            print 'created empty halos_results'
        else:
            halos_results = pyfits.getdata(filename_results_prob)
            print 'loaded ' , filename_results_prob

        global cfhtlens_shear_catalog
        if cfhtlens_shear_catalog == None:
            filename_chftlens_shears = os.environ['HOME']+ '/data/CFHTLens/CFHTLens_2014-04-07.fits'
            cfhtlens_shear_catalog = tabletools.loadTable(filename_chftlens_shears)
            if 'star_flag' in cfhtlens_shear_catalog.dtype.names:
                select = cfhtlens_shear_catalog['star_flag'] == 0
                cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
                select = cfhtlens_shear_catalog['fitclass'] == 0
                cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
                log.info('removed stars, remaining %d' , len(cfhtlens_shear_catalog))

                select = (cfhtlens_shear_catalog['e1'] != 0.0) * (cfhtlens_shear_catalog['e2'] != 0.0)
                cfhtlens_shear_catalog = cfhtlens_shear_catalog[select]
                log.info('removed zeroed shapes, remaining %d' , len(cfhtlens_shear_catalog))


        halo_z = vhalo['z']
        shear_bias_m = cfhtlens_shear_catalog['m']
        shear_weight = cfhtlens_shear_catalog['weight']

        # correcting additive systematics
        shear_g1 , shear_g2 = cfhtlens_shear_catalog['e1'] , -(cfhtlens_shear_catalog['e2']  - cfhtlens_shear_catalog['c2'])
        shear_ra_deg , shear_de_deg , shear_z = cfhtlens_shear_catalog['ra'] , cfhtlens_shear_catalog['dec'] ,  cfhtlens_shear_catalog['z']

        halo_ra_rad , halo_de_rad = cosmology.deg_to_rad(vhalo['ra'],vhalo['dec'])
        shear_ra_rad , shear_de_rad = cosmology.deg_to_rad(shear_ra_deg, shear_de_deg)

        # get tangent plane projection        
        shear_u_rad, shear_v_rad = cosmology.get_gnomonic_projection(shear_ra_rad , shear_de_rad , halo_ra_rad , halo_de_rad   )
        shear_g1_proj , shear_g2_proj = cosmology.get_gnomonic_projection_shear(shear_ra_rad , shear_de_rad , halo_ra_rad , halo_de_rad, shear_g1,shear_g2)       

        dtheta_x , dtheta_y = cosmology.arcmin_to_rad(box_size/2.,box_size/2.)
        select = ( np.abs( shear_u_rad ) < np.abs(dtheta_x)) * (np.abs(shear_v_rad) < np.abs(dtheta_y))

        shear_u_stamp_rad  = shear_u_rad[select]
        shear_v_stamp_rad  = shear_v_rad[select]
        shear_g1_stamp = shear_g1_proj[select]
        shear_g2_stamp = shear_g2_proj[select]
        shear_g1_orig = shear_g1[select]
        shear_g2_orig = shear_g2[select]
        shear_z_stamp = shear_z[select]
        shear_bias_m_stamp = shear_bias_m[select] 
        shear_weight_stamp = shear_weight[select]

        hist_g1, _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=shear_g1_stamp * shear_weight_stamp)
        hist_g2, _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=shear_g2_stamp * shear_weight_stamp)
        hist_n,  _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) )
        hist_m , _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=(1+shear_bias_m_stamp) * shear_weight_stamp )
        hist_w , _, _    = np.histogram2d( x=shear_u_stamp_rad, y=shear_v_stamp_rad , bins=(vec_u_rad,vec_v_rad) , weights=shear_weight_stamp)
        mean_g1 = hist_g1  / hist_w
        mean_g2 = hist_g2  / hist_w

        # in case we divided by zero
        hist_w[hist_n == 0] = 0
        hist_m[hist_n == 0] = 0
        hist_g1[hist_n == 0] = 0
        hist_g2[hist_n == 0] = 0
        mean_g1[hist_n == 0] = 0
        mean_g2[hist_n == 0] = 0

        select = np.isclose(hist_m,0)
        hist_w[select] = 0
        hist_m[select] = 0
        hist_g1[select] = 0
        hist_g2[select] = 0
        mean_g1[select] = 0
        mean_g2[select] = 0

        u_mid_arcmin, v_mid_arcmin = vec_u_arcmin[1:] - pixel_size/2. , vec_v_arcmin[1:] - pixel_size/2.
        grid_2d_u_arcmin , grid_2d_v_arcmin = np.meshgrid(u_mid_arcmin,v_mid_arcmin,indexing='ij')

        binned_u_arcmin = grid_2d_u_arcmin.flatten('F')
        binned_v_arcmin = grid_2d_v_arcmin.flatten('F')
        binned_g1 = mean_g1.flatten('F')
        binned_g2 = mean_g2.flatten('F')
        binned_n = hist_n.flatten('F')
        binned_w = hist_w.flatten('F')
 
        halo_shear = np.empty(len(binned_g1),dtype={'names':['u_arcmin','v_arcmin','g1','g2','weight','n_gals'],'formats':['f4']*5+['i4']})
        halo_shear['u_arcmin'] = binned_u_arcmin
        halo_shear['v_arcmin'] = binned_v_arcmin
        halo_shear['g1'] = binned_g1
        halo_shear['g2'] = binned_g2
        halo_shear['weight'] = binned_w
        halo_shear['n_gals'] = binned_n

        tabletools.savePickle(filename_shear,halo_shear,append=True,log=0)  

        import filaments_model_1h
        fitobj = filaments_model_1h.modelfit()
        fitobj.shear_u_arcmin =  halo_shear['u_arcmin']
        fitobj.shear_v_arcmin =  halo_shear['v_arcmin']
        fitobj.halo_u_arcmin = 0.
        fitobj.halo_v_arcmin = 0.
        fitobj.shear_g1 =  halo_shear['g1']
        fitobj.shear_g2 =  halo_shear['g2']
        fitobj.shear_w =  halo_shear['weight']
        fitobj.halo_z = vhalo['z']
        fitobj.get_bcc_pz(config['filename_pz'])
        fitobj.set_shear_sigma()
        fitobj.save_all_models=False

        fitobj.parameters[0]['box']['min'] = config['M200']['box']['min']
        fitobj.parameters[0]['box']['max'] = config['M200']['box']['max']
        fitobj.parameters[0]['n_grid'] = config['M200']['n_grid']

        log_post , grid_M200 = fitobj.run_gridsearch()
        ml_m200 = grid_M200[np.argmax(log_post)]
        
        halos_results['ihalo'][ih-id_first] = ihalo
        halos_results['m200_fit'][ih-id_first] = ml_m200
        halos_results['log_prob'][ih-id_first] = log_post

        pyfits.writeto(filename_results_prob,halos_results,clobber=True)
        print 'saved' , filename_results_prob

        log.info('%5d n_gals=%d n_eff=%2.2f m200_fit=%2.2f' % (ihalo,len(shear_g1_stamp),np.sum(shear_weight_stamp),ml_m200))



def main():

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    parser.add_argument('-f', '--first', default=0,type=int, action='store', help='first pair to process')
    parser.add_argument('-n', '--num', default=1,type=int, action='store', help='number of pairs to process')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')

    global args

    args = parser.parse_args()
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL, 
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]
    log.setLevel(logging_level)

    global config 
    config = yaml.load(open(args.filename_config))

    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    import filaments_cfhtlens
    # filaments_cfhtlens.select_halos_LRG(range_z=config['range_z'],range_M=config['range_M'],filename_halos=config['filename_allhalos_cat'],apply_graph=False)
    get_stamps()
    

    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


main()
