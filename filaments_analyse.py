import os
import matplotlib as mpl
if 'DISPLAY' not in os.environ:
    mpl.use('agg')
import os, yaml, argparse, sys, logging , pyfits, emcee, tabletools, cosmology, filaments_tools, plotstools, mathstools, scipy, scipy.stats
import numpy as np
import matplotlib.pyplot as pl
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


log = logging.getLogger("filam..fit") 
log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
log.addHandler(stream_handler)
log.propagate = False

cospars = cosmology.cosmoparams()

prob_z = None

dtype_stats = {'names' : ['id','kappa0_signif', 'kappa0_map', 'kappa0_err_hi', 'kappa0_err_lo', 'radius_map',    'radius_err_hi', 'radius_err_lo', 'chi2_red_null', 'chi2_red_max',  'chi2_red_D', 'chi2_red_LRT' , 'chi2_null', 'chi2_max', 'chi2_D' , 'chi2_LRT' ,'sigma_g' ] , 
        'formats' : ['i8'] + ['f8']*16 }

def get_prob_prod_mcmc(ids):

    n_per_file = 10
    id_file_first = args.first_result_file
    id_file_last = id_file_first + args.n_results_files
    n_params = 4
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','')

    prob_prod = []
    for ip in range(n_params): prob_prod.append(np.zeros(config['n_grid']))
    prob_kappa0_radius = np.zeros([config['n_grid_2D'],config['n_grid_2D']])
    
    n_missing=0
    n_usable_results=0
    ia=0

    filename_grid = 'results/results.grid.%s.pp2' % name_data
    grid_pickle = tabletools.loadPickle(filename_grid)
    grid = grid_pickle['list_params_marg']
    grid2D = grid_pickle['kappa0_radius_grid']

    grid2D_kappa0 = np.reshape(grid_pickle['kappa0_radius_grid'][:,0],[config['n_grid_2D'],config['n_grid_2D']])
    grid2D_radius = np.reshape(grid_pickle['kappa0_radius_grid'][:,1],[config['n_grid_2D'],config['n_grid_2D']])

    for nf in range(id_file_first,id_file_last):

        id_start = nf*n_per_file
        id_end = (nf+1)*n_per_file
        current_ids = range(id_start,id_end)

        if len(list(set(ids) & set(current_ids))) < 1:

            ia+=n_per_file
            log.debug('no requested results between %d and %d' , id_start, id_end)
            continue

        else:

            filename_pickle = 'results/results.chain.%04d.%04d.%s.pp2'  % (nf*n_per_file, (nf+1)*n_per_file , name_data)
            try:
                results_pickle = tabletools.loadPickle(filename_pickle,log=1)
            except:
                log.info('missing %s' % filename_pickle)
                n_missing +=1
                ia+=n_per_file
                continue

            for ni in range(n_per_file):

                if ia in ids:

                    # marginals in each parameter
                    for ip in range(n_params):

                        # turn this off
                        if ia>0: break

                        prob = results_pickle[ni]['list_prob_marg'][ip]
                        nans = np.nonzero(np.isnan(prob))[0]
                        if len(nans) > 0:
                            log.info('%d %s param=%d n_nans=%d', ni, filename_pickle , ip, len(np.nonzero(np.isnan(prob))[0]) )
                        prob[prob<1e-20]=1e-20
                        logprob = prob
                        # nans = np.nonzero(np.isnan( logprob))[0]
                        # infs = np.nonzero(np.isinf( logprob))[0]
                        # zeros = np.nonzero( logprob == 0.0)[0]

                        prod1D_pdf , prod1D_log_pdf , _ , _ = mathstools.get_normalisation(logprob)
                        try:
                            prob_prod[ip] += prod1D_log_pdf
                        except Exception,errmsg:
                            print errmsg
                            import pdb; pdb.set_trace()

                    # marginal kappa-radius
                    logprob2D = np.log(results_pickle[ni]['prob_kappa0_radius'])
                    # logprob2D[logprob2D<1e-20]=1e-20

                    # prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob2D)  
                    # nans = np.nonzero(np.isnan(prod2D_pdf))[0]
                    # n_nans=len(nans)
                    # if n_nans > 0: log.info('n_nans=%d',n_nans)
                    # prod2D_log_pdf = np.reshape(prod2D_log_pdf,[config['n_grid_2D'],config['n_grid_2D']])
                    
                    prod2D_log_pdf = np.reshape(logprob2D,[config['n_grid_2D'],config['n_grid_2D']])
                    prob_kappa0_radius += prod2D_log_pdf

                    # plot_prob_all, _, _,_ = mathstools.get_normalisation(prob_kappa0_radius)  
                    # plot_prob_this, _, _,_ = mathstools.get_normalisation(prod2D_log_pdf)                    
                    # pl.figure(figsize=(10,5))
                    # pl.subplot(1,2,1)
                    # pl.pcolormesh(grid2D_kappa0 , grid2D_radius , plot_prob_all); pl.colorbar()
                    # pl.xlim([-0.2,0.2])
                    # pl.ylim([-10,10])
                    # pl.subplot(1,2,2)
                    # pl.pcolormesh(grid2D_kappa0 , grid2D_radius , plot_prob_this); pl.colorbar()
                    # pl.xlim([-0.2,0.2])
                    # pl.ylim([-10,10])
                    # pl.show()

                    n_usable_results+=1
                ia+=1
                
            log.info('%4d %s n_usable_results=%d' , nf , filename_pickle , n_usable_results)

    for ip in range(n_params):
        prob1D_pdf , prod1D_log_pdf , _ , _ = mathstools.get_normalisation(prob_prod[ip])  
        prob_prod[ip] = prob1D_pdf
    prob1D_pdf = prob_prod

    prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(prob_kappa0_radius)  

        
    return prob1D_pdf, grid, prod2D_pdf, grid2D_kappa0, grid2D_radius, n_usable_results

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def get_prob_prod_gridsearch(ids,plots=False):

    n_per_file = 1
    id_file_first = args.first_result_file
    id_file_last = id_file_first + args.n_results_files
    n_params = 4
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')

    # initialise lists for 1D marginals
    list_pdf_prod_1D = []
    list_pdf_grid_1D = []

    
    n_missing=0
    n_usable_results=0

    filename_grid = '%s/results.grid.%s.pp2' % (args.results_dir,name_data)
    # filename_grid = 'results_local2scratch/results.grid.%s.pp2' % name_data
    grid_pickle = tabletools.loadPickle(filename_grid)
    grid_kappa0 = grid_pickle['grid_kappa0'][:,:,0,0]
    grid_radius = grid_pickle['grid_radius'][:,:,0,0]
    vec_kappa0 = grid_kappa0[:,0]
    vec_radius = grid_radius[0,:]

    n_upsample = 10

    # import pdb; pdb.set_trace()
    vec_kappa0_hires = np.linspace(min(grid_kappa0[:,0]),max(grid_kappa0[:,0]),len(grid_kappa0[:,0])*n_upsample)
    vec_radius_hires = np.linspace(min(grid_radius[0,:]),max(grid_radius[0,:]),len(grid_radius[0,:])*n_upsample)

    grid_kappa0_hires, grid_radius_hires = np.meshgrid(vec_kappa0_hires,vec_radius_hires,indexing='ij')
    
    logprob_kappa0_radius_hires = np.zeros([ len(grid_kappa0_hires) , len(grid_radius_hires) ])+66
    logprob_kappa0_radius = np.zeros([ len(grid_kappa0[:,0]) , len(grid_radius[0,:]) ])+66

    for nf in range(id_file_first,id_file_last):

        if nf in ids:

            # filename_pickle = 'results_local2scratch/results.prob.%04d.%04d.%s.pp2'  % (nf*n_per_file, (nf+1)*n_per_file , name_data)
            filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir,nf*n_per_file, (nf+1)*n_per_file , name_data)
            try:
                results_pickle = tabletools.loadPickle(filename_pickle,log=1)
            except:
                log.debug('missing %s' % filename_pickle)
                n_missing +=1
                continue
            if len(results_pickle) == 0:
                log.debug('empty %s' % filename_pickle)
                n_missing +=1
                continue

            # marginal kappa-radius
            log_prob = results_pickle
                               
            pdf_prob , _ , _ , _ = mathstools.get_normalisation(log_prob)  
            pdf_prob_2D = np.sum(pdf_prob,axis=(2,3))
            log_prob_2D = np.log(pdf_prob_2D)
            logprob_kappa0_radius += log_prob_2D
            plot_prob_all, _, _, _ = mathstools.get_normalisation(logprob_kappa0_radius)  
            plot_prob_this, _, _, _ = mathstools.get_normalisation(log_prob_2D)   

            # from scipy import interpolate
            # spline = interpolate.bisplrep(grid_kappa0,grid_radius,log_prob_2D,s=0)
            # log_prob_2D_hires = interpolate.bisplev(vec_kappa0_hires,vec_radius_hires,spline)
            # func_interp = interpolate.interp2d(grid_kappa0.T,grid_radius.T,log_prob_2D, kind='linear')
            # log_prob_2D_hires = func_interp(vec_kappa0_hires,vec_radius_hires)
            d1 =vec_kappa0[1]-vec_kappa0[0]
            d2 =vec_radius[1]-vec_radius[0]
            x=grid_kappa0_hires.flatten()/d1
            y=(grid_radius_hires.flatten() - min(grid_radius.flatten()))/d2
            log_prob_2D_hires = np.reshape(bilinear_interpolate(log_prob_2D,x,y),[len(vec_radius_hires),len(vec_radius_hires)]).T
            log_prob_2D_hires[-1,:] = 2*log_prob_2D_hires[-2,:] - log_prob_2D_hires[-3,:]
            log_prob_2D_hires[:,-1] = 2*log_prob_2D_hires[:,-2] - log_prob_2D_hires[:,-3]
            logprob_kappa0_radius_hires += log_prob_2D_hires

            n_usable_results+=1
            if plots:
                if nf % 100 == 0:
                    plot_prob_all_hires, _, _,_ = mathstools.get_normalisation(logprob_kappa0_radius_hires)  
                    plot_prob_this_hires, _, _,_ = mathstools.get_normalisation(log_prob_2D_hires)   
                    pl.figure(figsize=(10,10))
                    pl.subplot(2,2,1)
                    pl.pcolormesh(grid_kappa0, grid_radius , plot_prob_all); pl.colorbar()
                    pl.subplot(2,2,2)
                    pl.pcolormesh(grid_kappa0 , grid_radius , plot_prob_this); pl.colorbar()
                    pl.subplot(2,2,3)
                    pl.pcolormesh(grid_kappa0_hires, grid_radius_hires , plot_prob_all_hires); pl.colorbar()
                    pl.subplot(2,2,4)
                    pl.pcolormesh(grid_kappa0_hires , grid_radius_hires , plot_prob_this_hires); pl.colorbar()
                    pl.show()

            # now 1D
            # for ip in range(n_params):
            #     pdf_prod_1D
            #     list_pdf_prod_1D




            
            log.debug('%4d %s n_usable_results=%d' , nf , filename_pickle , n_usable_results)
            if nf % 100 == 0 : log.info('%4d n_usable_results=%d' , nf , n_usable_results)


    # prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob_kappa0_radius_hires)  
    # return None, None, prod2D_pdf, grid_kappa0_hires, grid_radius_hires, n_usable_results
    
    prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob_kappa0_radius)   
    return None, None, prod2D_pdf, grid_kappa0, grid_radius, n_usable_results
    
    # return None, None, prod2D_pdf, grid_kappa0, grid_radius, n_usable_results


def plot_vs_length():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    filename_pickle = args.filename_config.replace('.yaml','.plotdata.length.pp2')
    list_res_dict = tabletools.loadPickle(filename_pickle)
    nx=int(config['n_grid_2D']/2)
    # nx=0
  
    length = pairs['R_pair'] 

    pl.figure()
    pl.hist(length)

    for ib,bin in enumerate(list_res_dict):


        prod2D_pdf = list_res_dict[ib]['prod2D_pdf']
        grid2D_kappa0 = list_res_dict[ib]['grid2D_kappa0']
        grid2D_radius = list_res_dict[ib]['grid2D_radius']

        log.info('[%2.2e<mass<%2.2e]' % (bin['bin_min'] , bin['bin_max'] ))

        contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod2D_pdf)
        prod2D_pdf,_,_,_ = mathstools.get_normalisation(np.log(prod2D_pdf))

        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)
        pl.colorbar()
        # pl.xlim([0,0.3])
        # pl.ylim([0,2])
        pl.xlabel(r'$\Delta \Sigma 10^{14} M_{*} h \mathrm{Mpc}^{-2}$')
        pl.ylabel('half-mass radius [Mpc/h]')
        title_str= r'%s: filament length $\in [%2.1f , %2.1f]$ Mpc/h , n_pairs=%d' % (args.filename_config.replace('.yaml',''), bin['bin_min'] , bin['bin_max'] , bin['n_pairs_used'] )
        pl.title(title_str)
        pl.xlim([0,0.2])
        pl.ylim([0.25,4])
        filename_fig = 'figs/fig.length.%02d.%s.%d.png' % (ib,args.filename_config.replace('.yaml',''),bin['n_pairs_used'])
        pl.savefig(filename_fig)
        log.info('saved %s' % filename_fig)


        # pl.figure()
        # X = [grid2D_kappa0, grid2D_radius]
        # y = prod2D_pdf
        # plotstools.plot_dist_meshgrid(X,prod2D_pdf,contour=True,colormesh=True)
        # pl.subplot(2,2,1)
        # # pl.xlim([0,0.3])
        # pl.subplot(2,2,3)
        # # pl.xlim([0,0.3])
        # # pl.ylim([0,2])
        # pl.subplot(2,2,4)
        # pl.xlim([0,2])

        # grid2D_mass = grid2D_kappa0 /  (grid2D_radius * np.pi)
        # pl.figure()
        # pl.scatter(grid2D_mass.flatten(),grid2D_radius.flatten(),40,prod2D_pdf)
        # pl.xlim([0,0.1])
        # pl.ylim([0,2])
        # pl.xlabel("mass")
        # pl.ylabel('half-mass radius [Mpc]')
        # pl.title(str(bin['n_pairs_used']))


    pl.show()    


def plotdata_vs_length():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    bins_edges = [6,9,12,15,18]
    bins_centers = plotstools.get_bins_centers(bins_edges)

    list_res_dict = []

    for ib in range(1,len(bins_edges)):
        length = pairs['R_pair']
        # mass = (halo1['snr']+halo2['snr'])/2.
        ids = np.nonzero((length > bins_edges[ib-1]) * (length < bins_edges[ib]))[0]
        log.info('bin %d found n=%d ids' % (ib,len(ids)))
        list_prod_pdf , list_grid_pdf , prod2D_pdf ,  grid2D_kappa0 , grid2D_radius , n_pairs_used = get_prob_prod_gridsearch(ids)

        res_dict = {}
        res_dict['ib'] = ib
        res_dict['center'] = bins_centers[ib-1]
        res_dict['n_pairs_used'] = n_pairs_used
        res_dict['list_prod_pdf'] = list_prod_pdf
        res_dict['list_grid_pdf'] = list_grid_pdf
        res_dict['prod2D_pdf'] = prod2D_pdf
        res_dict['grid2D_kappa0'] = grid2D_kappa0
        res_dict['grid2D_radius'] = grid2D_radius
        res_dict['bin_min'] = bins_edges[ib-1]
        res_dict['bin_max'] = bins_edges[ib]

        list_res_dict.append(res_dict)

        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)

    filename_pickle = args.filename_config.replace('.yaml','.plotdata.length.pp2')
    tabletools.savePickle(filename_pickle,list_res_dict)

    pl.show()






def plotdata_vs_mass():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    n_pairs = len(halo1)

    if ('lrgclass' in filename_pairs) or ('clusterz' in filename_pairs):
        bins_snr_edges = [5e13,7.5e13,1e14,2e14,3e14]
        mass_param_name = 'm200'
        mass = (10**halo1[mass_param_name]+10**halo2[mass_param_name])/2.
    elif 'cfhtlens' in filename_pairs:
        bins_snr_edges = [0,6,20]
        mass_param_name = 'snr'
        mass = (halo1[mass_param_name]+halo2[mass_param_name])/2.
    if 'bcc' in filename_pairs: 
        bins_snr_edges = [1e14,2e14,3e14,4e14,5e14]
        mass_param_name = 'm200'
        mass = (halo1[mass_param_name]+halo2[mass_param_name])/2.

    # bins_snr_centers = [ 3 , 6]
    bins_snr_centers = plotstools.get_bins_centers(bins_snr_edges)

    # mass = np.max(np.concatenate([halo1[mass_param_name][:,None],halo2[mass_param_name][:,None]],axis=1),axis=1)
    # pl.hist(mass,bins=np.arange(10)-0.5); pl.show()

    list_res_dict = []

    for ib in range(1,len(bins_snr_edges)):
        # mass = (halo1[mass_param_name]+halo2[mass_param_name])/2.
        # mass = (halo1['snr']+halo2['snr'])/2.

        ids = np.nonzero((mass > bins_snr_edges[ib-1]) * (mass < bins_snr_edges[ib]))[0]

        log.info('bin %d: [%2.2e<mass<%2.2e], found n=%d ids' % (ib,  bins_snr_edges[ib-1], bins_snr_edges[ib], len(ids)))
        # if ib==1:
        list_prod_pdf , list_grid_pdf , prod2D_pdf ,  grid2D_kappa0 , grid2D_radius , n_pairs_used = get_prob_prod_gridsearch(ids)
        # if ib==2:
            # list_prod_pdf , list_grid_pdf , prod2D_pdf ,  grid2D_kappa0 , grid2D_radius , n_pairs_used = get_prob_prod_gridsearch(ids,plots=True)
        log.info('using %d pairs' , n_pairs_used)

        res_dict = {}
        res_dict['mass_param_name'] = mass_param_name
        res_dict['ib'] = ib
        res_dict['n_pairs_used'] = n_pairs_used
        res_dict['list_prod_pdf'] = list_prod_pdf
        res_dict['list_grid_pdf'] = list_grid_pdf
        res_dict['prod2D_pdf'] = prod2D_pdf
        res_dict['grid2D_kappa0'] = grid2D_kappa0
        res_dict['grid2D_radius'] = grid2D_radius
        res_dict['bin_min'] = bins_snr_edges[ib-1]
        res_dict['bin_max'] = bins_snr_edges[ib]

        list_res_dict.append(res_dict)



        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)
        pl.title('bin %d: [%2.2e<mass<%2.2e], n=%d ids' % (ib,  bins_snr_edges[ib-1], bins_snr_edges[ib], len(ids)))


    filename_pickle = args.filename_config.replace('.yaml','.plotdata.mass.pp2')
    tabletools.savePickle(filename_pickle,list_res_dict)

    pl.show()



def plot_vs_mass():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    filename_pickle = args.filename_config.replace('.yaml','.plotdata.mass.pp2')
    list_res_dict = tabletools.loadPickle(filename_pickle)
    nx=int(config['n_grid_2D']/2)

    if 'cfhtlens' in args.filename_config:
        mass_param_name = 'snr'
    if 'bcc' in args.filename_config:
        mass_param_name = 'm200'
    if 'lrgclass' in args.filename_config:
        mass_param_name = 'm200'

       
    for ib,snr_bin in enumerate(list_res_dict):

        log.info('[%2.2e<mass<%2.2e]' % (snr_bin['bin_min'] , snr_bin['bin_max'] ))

        prod2D_pdf = list_res_dict[ib]['prod2D_pdf']
        grid2D_kappa0 = list_res_dict[ib]['grid2D_kappa0']
        grid2D_radius = list_res_dict[ib]['grid2D_radius']

        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)

        contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod2D_pdf)
        # pl.colorbar()
        cp = pl.contour(grid2D_kappa0 , grid2D_radius, prod2D_pdf,levels=contour_levels,colors='m')
        # pl.clabel(cp, inline=1, fontsize=10

        pl.xlim([0,0.2])
        pl.ylim([0.25,4])
        # pl.xlabel("r'$\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}$'")
        pl.xlabel(r'$\Delta \Sigma 10^{14} M_{*} h \mathrm{Mpc}^{-2}$')
        pl.ylabel('half-mass radius [Mpc/h]')
        title_str= r'%s: mean halo %s $\in [%2.1e , %2.1e]$ , n_pairs=%d' % (args.filename_config.replace('.yaml',''),mass_param_name , snr_bin['bin_min'] , snr_bin['bin_max'] , snr_bin['n_pairs_used'] )
        pl.title(title_str)
        filename_fig = 'figs/fig.mass.%02d.%s.%d.png' % (ib,args.filename_config.replace('.yaml',''),snr_bin['n_pairs_used'])
        pl.savefig(filename_fig)
        log.info('saved %s' % filename_fig)

        # grid2D_mass = grid2D_kappa0 /  (grid2D_radius * np.pi)
        # pl.figure()
        # pl.scatter(grid2D_mass.flatten(),grid2D_radius.flatten(),40,prod2D_pdf)
        # pl.xlim([0,0.1])
        # pl.ylim([0,2])
        # pl.xlabel("mass")
        # pl.ylabel('half-mass radius [Mpc]')
        # pl.title(str(snr_bin['n_pairs_used']))


    pl.show()

def plotdata_all():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    n_pairs = len(halo1)

    if 'cfhtlens' in filename_pairs:
        bins_snr_edges = [5,20]
        mass_param_name = 'snr'
    else:
        bins_snr_edges = [1e14,1e15]
        mass_param_name = 'm200'
    # bins_snr_centers = [ 3 , 6]
    bins_snr_centers = plotstools.get_bins_centers(bins_snr_edges)

    list_res_dict = []

    ids=range(n_pairs)
    list_prod_pdf , list_grid_pdf , prod2D_pdf ,  grid2D_kappa0 , grid2D_radius , n_pairs_used = get_prob_prod_gridsearch(ids)

    pl.figure()
    pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)

    contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod2D_pdf)
    # pl.colorbar()
    cp = pl.contour(grid2D_kappa0 , grid2D_radius, prod2D_pdf,levels=contour_levels,colors='m')
    # pl.clabel(cp, inline=1, fontsize=10

    pl.xlim([0,0.3])
    pl.ylim([0,2])
    # pl.xlabel("r'$\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}$'")
    pl.xlabel("'\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}'")
    pl.ylabel('half-mass radius [Mpc]')
    title_str= "n_pairs=%d'" % (n_pairs_used)
    pl.title(title_str)
    filename_fig = 'figs/fig.all.%s.png' % (args.filename_config.replace('.yaml',''))
    pl.savefig(filename_fig)
    log.info('saved %s' % filename_fig)

    filename_pickle = args.filename_config.replace('.yaml','.plotdata.mass.pp2')
    tabletools.savePickle(filename_pickle,list_res_dict)

    pl.show()

def triangle_plots():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    id_file_first = args.first_result_file
    id_file_last = id_file_first + args.n_results_files

    filename_grid='%s/results.grid.%s.pp2' % (args.results_dir,os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits',''))
    grid=tabletools.loadPickle(filename_grid)
    X=[grid['grid_kappa0'],grid['grid_radius'],grid['grid_h1M200'],grid['grid_h2M200']]

    if halo2['m200'][0] > 100:
        halo1['m200'] = np.log10(halo1['m200'])
        halo2['m200'] = np.log10(halo2['m200'])

    res_all=None
    n_used=0

    mass= (halo1['m200']+halo2['m200'])/2.
    select = mass > 13

    for ida in range(id_file_first,id_file_last):

        if ~select[ida]:
            continue

        n_used+=1

        print "%d mean_mass=%2.2f halo1_mass=%2.2f halo2_mass=%2.2f z=%2.2f" % (ida,mass[ida],halo1['m200'][ida], halo2['m200'][ida] , halo2[ida]['z'])
        filename_result = '%s/results.prob.%04d.%04d.%s.pp2' % (args.results_dir,ida,ida+1,os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits',''))
        if os.path.isfile(filename_result):
            res=tabletools.loadPickle(filename_result,log=0)
        else:
            print 'missing' , filename_result
            continue

        if (res_all !=None):
            if (type(res)!=type(res_all)):
                print 'something wrong with the array', res
                continue
            if (res.shape != res_all.shape):
                print 'something wrong with the shape', res
                continue
        
        res_all = res if res_all == None else res_all+res

        # print halo1[ida]['m200'],halo2[ida]['m200']


    print 'median redshift z=%2.2f' % np.median(halo1['z'][select])
    print 'median mass m=%2.2f' % np.median(mass[select])
    print 'n_used', n_used

    import plotstools, mathstools
    prob=mathstools.normalise(res_all)
    plotstools.plot_dist_meshgrid(X,prob,labels=['kappa0','radius','m200_halo1','m200_halo2'])
    pl.show()


def plot_data_stamp():


    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    id_pair_first = args.first_result_file
    id_pair_last = args.first_result_file + args.n_results_files 

    for id_pair in range(id_pair_first,id_pair_last):
        

        if '.fits' in config['filename_shears']:
            shears_info = tabletools.loadTable(config['filename_shears'],hdu=id_pair+1)
        elif '.pp2' in config['filename_shears']:
            shears_info = tabletools.loadPickle(config['filename_shears'],pos=id_pair)

        filename_fig = 'figs/fig.stamp.%s.%05d.png' % (os.path.basename(config['filename_shears']).replace('.fits','').replace('.pp2',''),id_pair)
        
        pl.figure(figsize=(10,10))
        pl.subplot(3,1,1)
        filaments_tools.plot_pair(pairs['u1_mpc'][id_pair], pairs['v1_mpc'][id_pair], pairs['u2_mpc'][id_pair], pairs['v2_mpc'][id_pair], shears_info['u_mpc'], shears_info['v_mpc'], shears_info['g1'], shears_info['g2'],idp=id_pair,halo1=halo1[id_pair],halo2=halo2[id_pair],pair_info=pairs[id_pair],plot_type='quiver')
        pl.subplot(3,1,2)
        filaments_tools.plot_pair(pairs['u1_mpc'][id_pair], pairs['v1_mpc'][id_pair], pairs['u2_mpc'][id_pair], pairs['v2_mpc'][id_pair], shears_info['u_mpc'], shears_info['v_mpc'], shears_info['g1'], shears_info['g2'],idp=id_pair,halo1=halo1[id_pair],halo2=halo2[id_pair],pair_info=pairs[id_pair],plot_type='g1')
        pl.subplot(3,1,3)
        filaments_tools.plot_pair(pairs['u1_mpc'][id_pair], pairs['v1_mpc'][id_pair], pairs['u2_mpc'][id_pair], pairs['v2_mpc'][id_pair], shears_info['u_mpc'], shears_info['v_mpc'], shears_info['g1'], shears_info['g2'],idp=id_pair,halo1=halo1[id_pair],halo2=halo2[id_pair],pair_info=pairs[id_pair],plot_type='g2')
        pl.savefig(filename_fig)
        log.info('saved %s', filename_fig)
        pl.close()



   
def main():


    valid_actions = ['test_kde_methods', 'plot_vs_mass', 'plotdata_vs_mass' , 'plot_vs_length', 'plotdata_vs_length', 'plotdata_all' , 'triangle_plots', 'plot_data_stamp']

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-n', '--n_results_files',type=int,  default=1, action='store', help='number of results files to use')
    parser.add_argument('-f', '--first_result_file',type=int, default=0, action='store', help='number of first result file to open')
    parser.add_argument('-p', '--save_plots', action='store_true', help='if to save plots')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )
    parser.add_argument('-rd','--results_dir', action='store', help='where results files are' , default='results/' )

    # parser.add_argument('-d', '--dry', default=False,  action='store_true', help='Dry run, dont generate data')

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

    try:
        args.actions[0]
    except:
        raise Exception('choose one or more actions: %s' % str(valid_actions))

    for action in valid_actions:
        if action in args.actions:
            exec action+'()'
    for ac in args.actions:
        if ac not in valid_actions:
            print 'invalid action %s' % ac


    

main()
