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

def add_stats():

    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_halos']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halos = tabletools.loadTable(filename_halos)
    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)
    n_halos = len(halos)

    if not 'm200_h1_fit' in pairs.dtype.names:
        pairs = tabletools.appendColumn(arr=np.zeros(n_pairs),rec=pairs,name='m200_h1_fit',dtype='f4')
        pairs = tabletools.appendColumn(arr=np.zeros(n_pairs),rec=pairs,name='m200_h2_fit',dtype='f4')

    if not 'm200_fit' in halos.dtype.names:
        halos = tabletools.appendColumn(arr=np.zeros(n_halos),rec=halos,name='m200_fit',dtype='f4')

    filename_grid = '%s/results.grid.%s.pp2' % (args.results_dir,name_data)
    grid_dict = tabletools.loadPickle(filename_grid)

    id_file_first = args.first_result_file
    id_file_last = id_file_first + args.n_results_files
    
    n_missing=0
    for nf in range(id_file_first,id_file_last):

            filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir,nf, nf+1 , name_data)
            try:
                res = tabletools.loadPickle(filename_pickle,log=1)
            except:
                log.debug('missing %s' % filename_pickle)
                n_missing +=1
                continue
            if len(res) == 0:
                log.debug('empty %s' % filename_pickle)
                n_missing +=1
                continue

            prod_pdf = mathstools.normalise(res)
            grid_h1M200 = grid_dict['grid_h1M200'][0,0,:,0]
            grid_h2M200 = grid_dict['grid_h2M200'][0,0,0,:]

            prod_pdf_h1M200 = np.sum(prod_pdf,axis=(0,1,3))
            prod_pdf_h2M200 = np.sum(prod_pdf,axis=(0,1,2))

            # ml_mass_h1 = grid_h1M200[np.argmax(prod_pdf_h1M200)]
            # ml_mass_h2 = grid_h2M200[np.argmax(prod_pdf_h2M200)]
            ml_mass_h1 = grid_dict['grid_h1M200'].flatten()[np.argmax(prod_pdf)]
            ml_mass_h2 = grid_dict['grid_h2M200'].flatten()[np.argmax(prod_pdf)]

            pairs['m200_h1_fit'][nf] = ml_mass_h1
            pairs['m200_h2_fit'][nf] = ml_mass_h2

            ih1 = pairs['ih1'][nf]
            ih2 = pairs['ih2'][nf]
            halos['m200_fit'][ih1] = ml_mass_h1
            halos['m200_fit'][ih2] = ml_mass_h2

            log.info('%4d m200_h1_fit=%2.4f m200_h2_fit=%2.4f %d %d' % (nf,ml_mass_h1,ml_mass_h2,ih1,ih2) )

            # pl.plot(grid_h1M200,prod_pdf_h1M200,'r')
            # pl.plot(grid_h2M200,prod_pdf_h2M200,'g')
            # pl.axvline(ml_mass_h1,c='r')
            # pl.axvline(ml_mass_h2,c='g')
            # pl.axvline(halo1['m200'][nf],c='r',marker='o')
            # pl.axvline(halo2['m200'][nf],c='g',marker='o')          
            # pl.show()

    tabletools.saveTable(filename_pairs,pairs)
    tabletools.saveTable(filename_halos,halos)






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

def get_prob_prod_gridsearch(ids):

    id_file_first = args.first_result_file
    id_file_last = id_file_first + args.n_results_files
    n_params = 4
    name_data = os.path.basename(config['filename_shears']).replace('.pp2','').replace('.fits','')
  
    n_missing=0
    n_usable_results=0

    filename_grid = '%s/results.grid.%s.pp2' % (args.results_dir,name_data)
    grid_pickle = tabletools.loadPickle(filename_grid)

    res_all = None

    for nf in range(id_file_first,id_file_last):

        if nf in ids:

            filename_pickle = '%s/results.prob.%04d.%04d.%s.pp2'  % (args.results_dir,nf, nf+1 , name_data)
            try:
                res = tabletools.loadPickle(filename_pickle,log=1)
            except:
                log.debug('missing %s' % filename_pickle)
                n_missing +=1
                continue
            if len(res) == 0:
                log.debug('empty %s' % filename_pickle)
                n_missing +=1
                continue

            if (res_all !=None):
                if (type(res)!=type(res_all)):
                    print 'something wrong with the array', res
                    continue
                if (res.shape != res_all.shape):
                    print 'something wrong with the shape', res
                    continue

            res_all = res if res_all == None else res_all+res

            n_usable_results+=1
            log.debug('%4d %s n_usable_results=%d' , nf , filename_pickle , n_usable_results)
            if nf % 100 == 0 : log.info('%4d n_usable_results=%d' , nf , n_usable_results)


    prod_pdf = mathstools.normalise(res_all)   

    return prod_pdf, grid_pickle, n_usable_results
    


def get_prob_prod_gridsearch_2D(ids,plots=False,hires=True,hires_marg=False):
    
    import scipy.interpolate
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
    logprob_kappa0_radius = np.zeros([ len(grid_kappa0[:,0]) , len(grid_radius[0,:]) ])+66
    grid2D_dict = { 'grid_kappa0'  : grid_kappa0 , 'grid_radius' : grid_radius}  

    if hires:    
        n_upsample = 20
        vec_kappa0_hires = np.linspace(min(grid_kappa0[:,0]),max(grid_kappa0[:,0]),len(grid_kappa0[:,0])*n_upsample)
        vec_radius_hires = np.linspace(min(grid_radius[0,:]),max(grid_radius[0,:]),len(grid_radius[0,:])*n_upsample)
        grid_kappa0_hires, grid_radius_hires = np.meshgrid(vec_kappa0_hires,vec_radius_hires,indexing='ij')
        logprob_kappa0_radius_hires = np.zeros([ len(grid_kappa0_hires) , len(grid_radius_hires) ])+66

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
            # log_prob = results_pickle*214.524/2.577
            log_prob = results_pickle
            grid_h1M200 = grid_pickle['grid_h1M200'][0,0,:,0]
            grid_h2M200 = grid_pickle['grid_h2M200'][0,0,0,:]
            grid_h1M200_hires=np.linspace(grid_h1M200.min(),grid_h1M200.max(),len(grid_h1M200)*n_upsample)
            grid_h2M200_hires=np.linspace(grid_h2M200.min(),grid_h2M200.max(),len(grid_h2M200)*n_upsample)

            if hires_marg:

                log_prob_2D = np.zeros_like(log_prob[:,:,0,0])

                for i1 in range(len(log_prob[:,0,0,0])):
                    for i2 in range(len(log_prob[0,:,0,0])):
                        m200_2D = log_prob[i1,i2,:,:]
                        func_interp = scipy.interpolate.interp2d(grid_h1M200,grid_h2M200,m200_2D, kind='cubic')
                        m200_2D_hires = func_interp(grid_h1M200_hires,grid_h2M200_hires)
                        normalisation_const=2000
                        m200_2D_hires_prob=np.exp(m200_2D_hires-normalisation_const)
                        log_prob_2D[i1,i2] = np.log(np.sum(m200_2D_hires_prob))
                        
                        if plots:
                            print grid_kappa0[i1,i2], grid_radius[i1,i2] , np.sum(mathstools.normalise(m200_2D_hires))/len(m200_2D_hires) , np.sum(mathstools.normalise(m200_2D))/len(m200_2D)
                            pl.figure()
                            pl.subplot(1,2,1)
                            pl.imshow(mathstools.normalise(m200_2D),interpolation='nearest')
                            pl.subplot(1,2,2)
                            pl.imshow(mathstools.normalise(m200_2D_hires),interpolation='nearest')
                            pl.show()

            else:
                                   
                pdf_prob , _ , _ , _ = mathstools.get_normalisation(log_prob)  
                pdf_prob_2D = np.sum(pdf_prob,axis=(2,3))
                log_prob_2D = np.log(pdf_prob_2D)

            logprob_kappa0_radius += log_prob_2D
            plot_prob_all, _, _, _ = mathstools.get_normalisation(logprob_kappa0_radius)  
            plot_prob_this, _, _, _ = mathstools.get_normalisation(log_prob_2D)   

            if hires:
                # from scipy import interpolate
                # spline = interpolate.bisplrep(grid_kappa0,grid_radius,log_prob_2D,s=0)
                # log_prob_2D_hires = interpolate.bisplev(vec_kappa0_hires,vec_radius_hires,spline)
                func_interp = scipy.interpolate.interp2d(vec_kappa0,vec_radius,log_prob_2D, kind='cubic')
                log_prob_2D_hires = func_interp(vec_kappa0_hires,vec_radius_hires)
                # d1 =vec_kappa0[1]-vec_kappa0[0]
                # d2 =vec_radius[1]-vec_radius[0]
                # x=grid_kappa0_hires.flatten()/d1
                # y=(grid_radius_hires.flatten() - min(grid_radius.flatten()))/d2
                # log_prob_2D_hires = np.reshape(bilinear_interpolate(log_prob_2D,x,y),[len(vec_radius_hires),len(vec_radius_hires)]).T
                # log_prob_2D_hires[-1,:] = 2*log_prob_2D_hires[-2,:] - log_prob_2D_hires[-3,:]
                # log_prob_2D_hires[:,-1] = 2*log_prob_2D_hires[:,-2] - log_prob_2D_hires[:,-3]
                logprob_kappa0_radius_hires += log_prob_2D_hires

            if plots:
                if nf % 10 == 0:
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
                    pl.suptitle(nf)
                    pl.show()

          
            n_usable_results+=1
            log.debug('%4d %s n_usable_results=%d' , nf , filename_pickle , n_usable_results)
            if nf % 100 == 0 : log.info('%4d n_usable_results=%d' , nf , n_usable_results)


    if hires:
        grid2D_dict = { 'grid_kappa0'  : grid_kappa0_hires , 'grid_radius' : grid_radius_hires}  
        prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob_kappa0_radius_hires)  
        return prod2D_pdf, grid2D_dict, n_usable_results
    else:
        prod2D_pdf , prod2D_log_pdf , _ , _ = mathstools.get_normalisation(logprob_kappa0_radius)   
        return prod2D_pdf, grid2D_dict, n_usable_results
    
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

def plot_pickle(filename_pickle):

    # filename_pickle
    # filename_pickle = 'cfhtlens-lrgs.plotdata.mass.pp2'
    # filename_pickle = 'bcc-e.plotdata.mass.pp2'
    pickle=tabletools.loadPickle(filename_pickle)
    prob = pickle[0]['prob']
    grid = pickle[0]['params']
    # X=[grid['grid_kappa0'],grid['grid_radius'],grid['grid_h1M200'],grid['grid_h2M200']]
    X=[grid['grid_kappa0'],grid['grid_radius']]


    plotstools.plot_dist_meshgrid(X,prob,labels=[r"$\Delta\Sigma$  $10^{14} \mathrm{M}_{\odot} \mathrm{Mpc}^{-2} h$",r'radius $\mathrm{Mpc}/h$',r"M200_halo1 $\mathrm{M}_{\odot}/h$",r"M200_halo2 $\mathrm{M}_{\odot}/h$"],contour=True)
    # pl.suptitle('CFHTLens + BOSS-DR10 LRGs, using %d halo pairs' % pickle[0]['n_obj'])
    pl.suptitle('CFHTLens + BOSS-DR10, using %d halo pairs' % pickle[0]['n_obj'])
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
    filename_pairs = config['filename_pairs'].replace('.fits','.addstats.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
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
    
    # filename_prune = 'pairs_cfhtlens_lrgs.prune.fits'
    # pairs_prune = tabletools.loadTable(filename_prune)
    # select_prune = np.array([ (True if pairs['ipair'][ip] in pairs_prune['ipair'] else False) for ip in pairs['ipair']])

    mass= (pairs['m200_h1_fit']+pairs['m200_h2_fit'])/2.
    select = (mass < 16.5) * (mass > 13.5)

    # ids=np.arange(n_pairs)[select*select_prune]
    ids=np.arange(n_pairs)[select]
    # prod_pdf, grid_dict, n_pairs_used = get_prob_prod_gridsearch(ids)
    prod_pdf, grid_dict, n_pairs_used = get_prob_prod_gridsearch_2D(ids)


    res_dict = { 'prob' : prod_pdf , 'params' : grid_dict, 'n_obj' : n_pairs_used }

    list_res_dict.append(res_dict)
    filename_pickle = args.filename_config.replace('.yaml','.plotdata.mass.pp2')
    tabletools.savePickle(filename_pickle,list_res_dict)

    plot_pickle(filename_pickle)

    # grid_kappa0 = grid_dict['grid_kappa0'][:,:,0,0]
    # grid_radius = grid_dict['grid_radius'][:,:,0,0]
    # prod_pdf_kappa0_radius = np.sum(prod_pdf,axis=(2,3))

    # pl.figure()
    # pl.pcolormesh( grid_kappa0 , grid_radius, prod_pdf_kappa0_radius)

    # contour_levels , contour_sigmas = mathstools.get_sigma_contours_levels(prod_pdf_kappa0_radius)
    # # pl.colorbar()
    # cp = pl.contour(grid_kappa0 , grid_radius, prod_pdf_kappa0_radius,levels=contour_levels,colors='m')
    # # pl.clabel(cp, inline=1, fontsize=10

    # pl.xlim([0,0.3])
    # pl.ylim([0,2])
    # # pl.xlabel("r'$\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}$'")
    # pl.xlabel("'\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}'")
    # pl.ylabel('half-mass radius [Mpc]')
    # pl.axis('tight')
    # title_str= "n_pairs=%d" % (n_pairs_used)
    # pl.title(title_str)
    # filename_fig = 'figs/fig.all.%s.png' % (args.filename_config.replace('.yaml',''))
    # pl.savefig(filename_fig)
    # log.info('saved %s' % filename_fig)
    # pl.show()

def triangle_plots():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')
    filename_pairs = config['filename_pairs'].replace('.fits','.addstats.fits')

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

    mass= (pairs['m200_h1_fit']+pairs['m200_h2_fit'])/2.
    select = mass > 13.

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

    print 'median redshift z=%2.2f' % np.median(halo1['z'][select])
    print 'median mass m=%2.2f' % np.median(mass[select])
    print 'n_used', n_used

    import plotstools, mathstools
    prob=mathstools.normalise(res_all)

    labels=[r"$\Delta \Sigma 10^{14} * M_{\odot} \mathrm{Mpc}^{-2} h$",'radius Mpc/h',r"$M200_halo1 M_{\odot}/h$",r"$M200_halo2 M_{\odot}/h$"]
    mdd = plotstools.multi_dim_dist()
    mdd.labels=labels
    mdd.plot_dist_meshgrid(X,prob)
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


def plot_halo_map():

    filaments_tools.get_halo_map(config['filename_pairs'])

def remove_similar_connections():

    filename_pairs = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    pairs['ipair'] = np.arange(len(pairs))

    n_pairs = len(pairs)
    remove_list = []
    pairs_new = range(len(pairs))
    min_angle = 30 # deg
    min_dist=4
    n_all = 0

    unique_halos, unique_indices = np.unique(np.concatenate([pairs['ih1'],pairs['ih2']]),return_index=True)
    unique_ra = np.concatenate([halo1,halo2])['ra'][unique_indices]
    unique_de = np.concatenate([halo1,halo2])['dec'][unique_indices]
    unique_z = np.concatenate([halo1,halo2])['z'][unique_indices]
    unique_DA =  cosmology.get_ang_diam_dist(unique_z)  
    unique_haloinfo = np.concatenate([halo1,halo2])[unique_indices]
    x,y,z = cosmology.spherical_to_cartesian_with_redshift(unique_ra,unique_de,unique_z)
    n_unique=len(unique_de)
    box_coords = np.concatenate([x[:,None],y[:,None],z[:,None]],axis=1)

    log.info('getting Ball Tree for 3D')
    BT = BallTree(box_coords, leaf_size=5)
    n_connections=100
    bt_dx,bt_id = BT.query(box_coords,k=n_connections)

    bt_id_reduced = bt_id[:,1:]
    bt_dx_reduced = bt_dx[:,1:]
    ih1 = np.kron( np.ones((n_connections-1,1),dtype=np.int8), np.arange(0,n_unique) ).T
    ih2 = bt_id_reduced
    DA  = bt_dx_reduced

    log.info('number of pairs %d ' % len(ih1))
    select = ih1 < ih2
    ih1 = ih1[select]
    ih2 = ih2[select]
    DA = DA[select]
    log.info('number of pairs after removing duplicates %d ' % len(ih1))

    log.info('calculating x-y distance')
    halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(  unique_ra[ih1] , unique_de[ih1]  )
    halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(  unique_ra[ih2] , unique_de[ih2]  )
    
    min_dist = 1
    # d_xy  = (unique_DA[ih1] + unique_DA[ih2])/2. * cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad)
    d_xy  =  cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad)*180./np.pi

    remove_lrgs = []
    for ip in range(len(d_xy)):
        if ip >= len(d_xy):
            continue
        if d_xy[ip] < min_dist:

            print ih1[ip],ih2[ip],d_xy[ip], unique_haloinfo['dered_r'][ih1[ip]] , unique_haloinfo['dered_r'][ih2[ip]], len(ih1), len(ih2), len(d_xy)
            if unique_haloinfo['dered_r'][ih1[ip]] < unique_haloinfo['dered_r'][ih2[ip]]:
                remove_lrgs.append(unique_halos[ih2[ip]])
                select = (ih1 == ih2[ip]) | (ih2 == ih2[ip])
                ih1  = ih1[~select]
                ih2  = ih2[~select]
                d_xy = d_xy[~select]
            else:
                remove_lrgs.append(unique_halos[ih1[ip]])
                select = (ih1 == ih1[ip]) | (ih2 == ih1[ip])
                ih1  = ih1[~select]
                ih2  = ih2[~select]
                d_xy = d_xy[~select]
            print len(ih1), len(ih2), len(d_xy)
            
    print len(remove_lrgs), len(unique_halos)

    pl.figure()
    pl.scatter(unique_ra,unique_de,c='r')
    pl.scatter(unique_ra[ih1],unique_de[ih1],c='b')
    pl.scatter(unique_ra[ih2],unique_de[ih2],c='c')
    pl.show()

    list_pairs = []
    for ip, vp in enumerate(pairs):
        if (vp['ih1'] in remove_lrgs) or (vp['ih2'] in remove_lrgs):
            remove_list.append(ip)

    for x in set(remove_list): pairs_new.remove(x)
    pairs_prune = pairs[pairs_new]
    halo1_prune = halo1[pairs_new]
    halo2_prune = halo2[pairs_new]

    mass= (pairs_prune['m200_h1_fit'] + pairs_prune['m200_h2_fit'])/2.
    sorting = np.argsort(mass)[::-1]

    pairs_prune = pairs_prune[sorting]
    halo1_prune = halo1_prune[sorting]
    halo2_prune = halo2_prune[sorting]

    filename_pairs_selected = filename_pairs.replace('.fits','.prune.fits')
    filename_halo1_selected = filename_halos1.replace('.halos1.fits','.prune.halos1.fits')
    filename_halo2_selected = filename_halos2.replace('.halos2.fits','.prune.halos2.fits')
    tabletools.saveTable(filename_pairs_selected,pairs_prune)
    tabletools.saveTable(filename_halo1_selected, halo1_prune)
    tabletools.saveTable(filename_halo2_selected, halo2_prune)

    # filaments_tools.get_halo_map(filename_pairs,color='c')
    # filaments_tools.get_halo_map(filename_pairs_selected,color='r')
    # pl.show()

    pairs = pairs_prune
    halo1 = halo1_prune
    halo2 = halo2_prune
    remove_list = []
    pairs_new = range(len(pairs))

    for ic1,vc1 in enumerate(pairs):
        for ic2,vc2 in enumerate(pairs):

            # only one triangle:
            if ic1>ic2:
        
                # connection 1 and 2 have the same node
                same_node=None
                if vc1['ih1'] == vc2['ih1']:
                    same_node_ra , same_node_de = vc2['ra1'] ,  vc2['dec1']
                    other1_ra , other1_de = vc1['ra2'] , vc1['dec2'] 
                    other2_ra , other2_de = vc2['ra2'] , vc2['dec2']
                elif vc1['ih2'] == vc2['ih2']:
                    same_node_ra , same_node_de = vc2['ra2'] ,  vc2['dec2']
                    other1_ra , other1_de = vc1['ra1'] , vc1['dec1'] 
                    other2_ra , other2_de = vc2['ra1'] , vc2['dec1']
                elif vc1['ih1'] == vc2['ih2']:
                    same_node_ra , same_node_de = vc2['ra1'] ,  vc2['dec1']
                    other1_ra , other1_de = vc1['ra2'] , vc1['dec2'] 
                    other2_ra , other2_de = vc2['ra2'] , vc2['dec2']
                else:
                    continue

                # decide if to remove connection or not

                x1=np.array( [other1_ra , other1_de] ) - np.array( [ same_node_ra , same_node_de ] )
                x2=np.array( [other2_ra , other2_de] ) - np.array( [ same_node_ra , same_node_de ] )
                angle = np.arccos( np.dot(x1,x2) / np.linalg.norm(x1) / np.linalg.norm(x2) ) /np.pi * 180

                if angle > min_angle:
                    # print '%4d %4d %4d %4d %5.4f  - saved both %d %d' % (vc1['ih1'],vc1['ih2'],vc2['ih1'],vc2['ih2'],angle,vc1['ipair'],vc2['ipair']) 
                    pass
                else:
                    mass1= vc1['m200_h1_fit'] + vc1['m200_h2_fit']
                    mass2= vc2['m200_h1_fit'] + vc2['m200_h2_fit']
                    if mass1>=mass2:
                        remove_list.append(vc2['ipair'])
                        # print 'removed' , vc1['ipair']
                    else:
                        remove_list.append(vc1['ipair'])
                    n_all+=1
                    if n_all % 100 == 0 : print n_all, n_pairs**2/2., len(remove_list), len(set(remove_list))
                        # print 'removed' , vc2['ipair']

    for ip,vp in enumerate(pairs): 
        if vp['ipair'] in remove_list:
            pairs_new.remove(ip)

    pairs_prune = pairs[pairs_new]
    halo1_prune = halo1[pairs_new]
    halo2_prune = halo2[pairs_new]

    mass= (pairs_prune['m200_h1_fit'] + pairs_prune['m200_h2_fit'])/2.
    sorting = np.argsort(mass)[::-1]

    pairs_prune = pairs_prune[sorting]
    halo1_prune = halo1_prune[sorting]
    halo2_prune = halo2_prune[sorting]

    filename_pairs_selected = filename_pairs.replace('.fits','.prune2.fits')
    filename_halo1_selected = filename_halos1.replace('.halos1.fits','.prune2.halos1.fits')
    filename_halo2_selected = filename_halos2.replace('.halos2.fits','.prune2.halos2.fits')
    tabletools.saveTable(filename_pairs_selected,pairs_prune)
    tabletools.saveTable(filename_halo1_selected, halo1_prune)
    tabletools.saveTable(filename_halo2_selected, halo2_prune)

    filaments_tools.get_halo_map(filename_pairs,color='c')
    filaments_tools.get_halo_map(filename_pairs_selected,color='r')
    pl.show()

   
def main():


    valid_actions = ['test_kde_methods', 'plot_vs_mass', 'plotdata_vs_mass' , 'plot_vs_length', 'plotdata_vs_length', 'plotdata_all' , 'triangle_plots', 'plot_data_stamp','add_stats' ,'plot_halo_map' , 'plot_pickle','remove_similar_connections']

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

    # parser.add_argument('-d', '--dry', default=False,  action='store_true', help='Dry run, dont generate data'), len(remove_list), len(set(remove_list)

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
