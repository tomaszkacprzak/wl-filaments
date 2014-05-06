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
try:
    from pyqt_fit import kde
except:
    print 'loading pyqt_fit failed'
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

from guppy import hpy
h = hpy()
dtype_stats = {'names' : ['id','kappa0_signif', 'kappa0_map', 'kappa0_err_hi', 'kappa0_err_lo', 'radius_map',    'radius_err_hi', 'radius_err_lo', 'chi2_red_null', 'chi2_red_max',  'chi2_red_D', 'chi2_red_LRT' , 'chi2_null', 'chi2_max', 'chi2_D' , 'chi2_LRT' ,'sigma_g' ] , 
        'formats' : ['i8'] + ['f8']*16 }


def analyse_stats():

    filename_results_stats = 'results.stats.shears_bcc_e.cat'
    stats = tabletools.loadTable(filename_results_stats,dtype=dtype_stats)

    # plotstools.plot_dist( np.array([ stats['kappa0_map'] , stats['radius_map'] ]) )
    # pl.show()
    pl.figure()
    pl.subplot(121)
    pl.hist(stats['kappa0_map'],bins=50)
    pl.xlabel('kappa0 ML')
    pl.subplot(122)
    pl.hist(stats['radius_map'],bins=50)
    pl.xlabel('radius ML')

    filename_fig = 'figs/stats.histograms.png'
    pl.savefig(filename_fig)
    log.info('saved %s' , filename_fig)

    print 'fraction of filaments with kappa > 0.0005'
    select_k = (stats['kappa0_map'] > 0.0005) * (stats['kappa0_map'] < 0.025)
    print sum(select_k) 

    print 'fraction of filaments with  0.5 > radius < 8 '
    select_r = (stats['radius_map'] < 8) * (stats['radius_map'] > 0.25)
    print sum(select_r) 

    print 'fraction of filaments with valid kappa and radius'
    print sum( select_r * select_k )           

    valid_stats = stats[select_r * select_k]

    import pdb; pdb.set_trace()
    kappa0_mean = np.mean(valid_stats['kappa0_map']) 
    kappa0_err = ( valid_stats['kappa0_err_hi'] + valid_stats['kappa0_err_lo'] ) / 2. 
    kappa0_stdm = np.sqrt( sum(kappa0_err**2) ) / len(kappa0_err)

    print kappa0_mean , kappa0_stdm

    pl.figure()
    pl.scatter(stats['kappa0_map'][select_r],stats['radius_map'][select_r],s=20,c='r',marker='o')
    pl.scatter(stats['kappa0_map'][select_k],stats['radius_map'][select_k],s=20,c='g',marker='o')
    pl.scatter(stats['kappa0_map'][select_r * select_k],stats['radius_map'][select_r * select_k],s=20,c='b',marker='o')
    pl.xlabel('kappa0')
    pl.ylabel('radius')
    filename_fig = 'figs/stats.scatter_radius_kappa0.png'
    pl.savefig(filename_fig)
    log.info('saved %s' , filename_fig)
    # pl.show()

    filename_results_grid= 'results.grid.pp2'
    grid = tabletools.loadPickle(filename_results_grid)
    import pdb; pdb.set_trace()


    filename_results_pdfs= 'results.prob.shears_bcc_e.pp2'
    pdfs = np.array(tabletools.loadPickle(filename_results_pdfs))
    pdfs_sum = np.sum(pdfs[select_r * select_k],axis=0)

    log_pdfs = np.log(pdfs)

    pdfs_prod = np.sum(log_pdfs[select_r * select_k],axis=0)
    pdfs_prod = np.exp(pdfs_prod - max(pdfs_prod)) 
    pdfs_prod = pdfs_prod / sum(pdfs_prod)

    n_grid = int(np.sqrt(len(pdfs[0])))
    import pdb; pdb.set_trace()
    
    prob_post_matrix = np.reshape(  pdfs_prod , [n_grid, n_grid] )    
    prob_post_kappa0 = prob_post_matrix.sum(axis=1)
    prob_post_radius = prob_post_matrix.sum(axis=0)

    # pl.imshow(prob_post_matrix)
    # pl.show()

    pl.figure()
    plotstools.imshow_grid(grid['kappa0_post'],grid['radius_post'],pdfs_prod)
    pl.show()


    print pdfs.shape

    import pdb; pdb.set_trace()


    pass    

# def analyse_results():

#     filename_prob_array = 'results.prob.shears_bcc_g.pp2'
#     res_prob_array = tabletools.loadPickle(filename_prob_array)[0]
#     n_pairs = res_prob_array.shape[0]

#     filename_grid = 'results.grid.bcc.pp2'
#     file_grid = open(filename_grid,'r')
#     import cPickle as pickle

#     grid=pickle.load(file_grid)
#     # ['grid_radius', 'grid_kappa0', 'radius_post', 'kappa0_post']
#     n_grid = grid['grid_kappa0'].shape[0]
    
#     # prob_prod = np.prod(res_prob_array,axis=0)

#     # prob_post_matrix = np.reshape(  prob_prod , [n_grid, n_grid] )    
#     # prob_post_kappa0 = prob_post_matrix.sum(axis=1)
#     # prob_post_radius = prob_post_matrix.sum(axis=0)

#     # pl.imshow(prob_post_matrix)
#     # pl.colorbar()
#     # pl.show()

#     filename_pairs = 'results.pairs.bcc.cat'
#     pairs_results = tabletools.loadTable(filename_pairs,dtype=dtype_stats)
    

#     results_kappa0 = pairs_results['kappa0_err_lo'] # bug
#     results_radius = pairs_results['radius_map']
#     results_chi2_LRT =  pairs_results['chi2_LRT']

#     pl.subplot(2,2,1)
#     pl.scatter(results_kappa0, results_radius)
#     pl.xlabel('kappa0 max likelihood')
#     pl.ylabel('radius max likelihood')
    
#     pl.subplot(2,2,2)
#     pl.scatter(results_kappa0,results_chi2_LRT)
#     pl.xlabel('kappa0 max likelihood')
#     pl.ylabel('chi2 p-val')

#     pl.show()

#     import pdb; pdb.set_trace()


# def process_results():

#     filename_prob = 'results.prob.shears_bcc_g.pp2'
#     filename_prob_array = filename_prob.replace('.prob.', '.prob-array.')
#     file_grid = open(filename_prob,'r')
#     filename_grid = 'results.grid.bcc.pp2'
#     file_prob = open(filename_prob,'r')
#     n_pairs = 3000

#     import cPickle as pickle

#     grid=pickle.load(file_grid)
#     n_grid = len(grid['kappa0_post'])

#     prob_post_array = np.zeros([n_pairs,n_grid],dtype=np.float32)

#     for n in range(n_pairs):
#         res=pickle.load(file_prob)
#         prob_post_array[n,:] = res['prob_post']
#         print n

#     print prob_post_array.shape

#     tabletools.savePickle(filename_prob_array, prob_post_array)

#     # prob_post_matrix = np.reshape(  prob_post , [n_grid, n_grid] )    
#     # prob_post_kappa0 = prob_post_matrix.sum(axis=1)
#     # prob_post_radius = prob_post_matrix.sum(axis=0)


#     pl.imshow(prob_post_array)
#     pl.show()

def analyse_stats_samples():

    filename_results_stats = 'results.stats.0048.0050.shears_bcc_g.cat'
    stats = tabletools.loadTable(filename_results_stats,dtype=filaments_model_2hf.dtype_stats)

    # plotstools.plot_dist( np.array([ stats['kappa0_map'] , stats['radius_map'] ]) )
    # pl.show()
    pl.figure()
    pl.subplot(121)
    pl.hist(stats['kappa0_map'],bins=50)
    pl.xlabel('kappa0 ML')
    pl.subplot(122)
    pl.hist(stats['radius_map'],bins=50)
    pl.xlabel('radius ML')

    filename_fig = 'figs/stats.histograms.png'
    pl.savefig(filename_fig)
    log.info('saved %s' , filename_fig)

    print 'fraction of filaments with kappa > 0.0005'
    select_k = (stats['kappa0_map'] > 0.0005) * (stats['kappa0_map'] < 0.025)
    print sum(select_k) 

    print 'fraction of filaments with  0.5 > radius < 8 '
    select_r = (stats['radius_map'] < 8) * (stats['radius_map'] > 0.25)
    print sum(select_r) 

    print 'fraction of filaments with valid kappa and radius'
    print sum( select_r * select_k )           

    valid_stats = stats[select_r * select_k]

    import pdb; pdb.set_trace()
    kappa0_mean = np.mean(valid_stats['kappa0_map']) 
    kappa0_err = ( valid_stats['kappa0_err_hi'] + valid_stats['kappa0_err_lo'] ) / 2. 
    kappa0_stdm = np.sqrt( sum(kappa0_err**2) ) / len(kappa0_err)

    print kappa0_mean , kappa0_stdm

    pl.figure()
    pl.scatter(stats['kappa0_map'][select_r],stats['radius_map'][select_r],s=20,c='r',marker='o')
    pl.scatter(stats['kappa0_map'][select_k],stats['radius_map'][select_k],s=20,c='g',marker='o')
    pl.scatter(stats['kappa0_map'][select_r * select_k],stats['radius_map'][select_r * select_k],s=20,c='b',marker='o')
    pl.xlabel('kappa0')
    pl.ylabel('radius')
    filename_fig = 'figs/stats.scatter_radius_kappa0.png'
    pl.savefig(filename_fig)
    log.info('saved %s' , filename_fig)
    # pl.show()

    filename_results_pdfs= 'results.chain.0048.0050.shears_bcc_g.pp2'
    results_struct = np.array(tabletools.loadPickle(filename_results_pdfs))
    
    for i in range(4):
        pl.figure()
        pl.plot(results_struct[0]['list_params_marg'][i] , results_struct[0]['list_prob_marg'][i])

    pl.show()

    import pdb; pdb.set_trace()
    
    pdfs_sum = np.sum(pdfs[select_r * select_k],axis=0)

    log_pdfs = np.log(pdfs)

    pdfs_prod = np.sum(log_pdfs[select_r * select_k],axis=0)
    pdfs_prod = np.exp(pdfs_prod - max(pdfs_prod)) 
    pdfs_prod = pdfs_prod / sum(pdfs_prod)

    n_grid = int(np.sqrt(len(pdfs[0])))
    import pdb; pdb.set_trace()
    
    prob_post_matrix = np.reshape(  pdfs_prod , [n_grid, n_grid] )    
    prob_post_kappa0 = prob_post_matrix.sum(axis=1)
    prob_post_radius = prob_post_matrix.sum(axis=0)

    # pl.imshow(prob_post_matrix)
    # pl.show()

    pl.figure()
    plotstools.imshow_grid(grid['kappa0_post'],grid['radius_post'],pdfs_prod)
    pl.show()


    print pdfs.shape

    import pdb; pdb.set_trace()


    pass    


    
def test_kde_methods():

    name_data = os.path.basename(config['filename_shears']).replace('.fits','')
    filename_results_cat = 'results.stats.%s.cat' % name_data
    stats = tabletools.loadTable(filename_results_cat,dtype=filaments_model_2hf.dtype_stats)

    n_per_file = 10
    n_files = args.n_results_files
    log.info('n_files=%d',n_files)

    list_DeltaSigma = []

    n_colors = 10
    ic =0 
    n_grid = 10001
    grid_DeltaSigma_edges = np.linspace(0,1,n_grid)
    grid_DeltaSigma_centers = plotstools.get_bins_centers(grid_DeltaSigma_edges)
    pixel_size = grid_DeltaSigma_centers[1] - grid_DeltaSigma_centers[0]
    list_ids = []

    ia=0
    for nf in range(n_files):
    # for nf in range(2):

        filename_pickle = 'results/results.chain.%04d.%04d.%s.pp2'  % (nf*n_per_file, (nf+1)*n_per_file , name_data)
        try:
            results_pickle = tabletools.loadPickle(filename_pickle)
            log.info('%4d %s' , nf , filename_pickle)
        except:
            log.info('missing %s' % filename_pickle)
            ia+=1
            continue

        for ni in range(n_per_file):

            # prob_DeltaSigma_kde = results_pickle[ni]['list_prob_marg'][0]
            # grid_DeltaSigma_kde = results_pickle[ni]['list_params_marg'][0]

            chain = results_pickle[ni]['flatchain'][0][:,0]
            est_ref = kde.KDE1D(chain, lower=0 , upper=1, method='reflexion')           
            est_lin = kde.KDE1D(chain, lower=0 , upper=1, method='linear_combination')                   
            prob_DeltaSigma_kde_lin = mathstools.get_func_split(func=est_ref,grid=grid_DeltaSigma_centers)
            prob_DeltaSigma_kde_ref = mathstools.get_func_split(func=est_ref,grid=grid_DeltaSigma_centers)          
            prob_DeltaSigma_hist , _ = np.histogram(results_pickle[ni]['flatchain'][0][:,0], bins=grid_DeltaSigma_edges,normed=True)

            prob_DeltaSigma_kde_ref /= np.sum(prob_DeltaSigma_kde_ref)
            prob_DeltaSigma_kde_lin /= np.sum(prob_DeltaSigma_kde_lin)
            prob_DeltaSigma_hist /= np.sum(prob_DeltaSigma_hist)

            pl.plot(grid_DeltaSigma_centers,prob_DeltaSigma_kde_ref ,'b',label='ref')
            pl.plot(grid_DeltaSigma_centers,prob_DeltaSigma_kde_lin ,'g',label='lin')
            pl.plot(grid_DeltaSigma_centers,prob_DeltaSigma_hist,'rd')
            pl.xlim([0,0.1])
            pl.legend()
            pl.show()    

def get_prob_prod(ids):

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



def plot_vs_length():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    pairs = tabletools.loadTable(filename_pairs)
    n_pairs = len(halo1)

    filename_pickle = 'plotdata.length.pp2'
    list_res_dict = tabletools.loadPickle(filename_pickle)
    nx=int(config['n_grid_2D']/2)
    # nx=0
  
    length = pairs['R_pair'] 

    pl.figure()
    pl.hist(length)

    for bin in list_res_dict:

        grid2D_kappa0 = bin['grid2D_kappa0'][nx:,nx:]
        grid2D_radius = bin['grid2D_radius'][nx:,nx:]       
        prod2D_pdf = bin['prod2D_pdf'][nx:,nx:]

        # grid2D_kappa0 = bin['grid2D_kappa0'][:(nx+1),nx:]
        # grid2D_radius = bin['grid2D_radius'][:(nx+1),nx:]       
        # prod2D_pdf = bin['prod2D_pdf'][:(nx+1),nx:]

        # grid2D_kappa0 = bin['grid2D_kappa0']
        # grid2D_radius = bin['grid2D_radius']
        # prod2D_pdf = bin['prod2D_pdf']

        import pdb; pdb.set_trace()

        prod2D_pdf,_,_,_ = mathstools.get_normalisation(np.log(prod2D_pdf))

        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)
        pl.colorbar()
        # pl.xlim([0,0.3])
        # pl.ylim([0,2])
        pl.xlabel("r'$\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}$'")
        pl.ylabel('half-mass radius [Mpc]')
        pl.title('bin length=%2.2f n_pairs=%d' % ( bin['center'] , bin['n_pairs_used']) )

        pl.figure()
        X = [grid2D_kappa0, grid2D_radius]
        y = prod2D_pdf
        plotstools.plot_dist_meshgrid(X,prod2D_pdf,contour=True,colormesh=True)
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

    bins_edges = [6,18]
    bins_centers = plotstools.get_bins_centers(bins_edges)

    list_res_dict = []

    for ib in range(1,len(bins_edges)):
        length = pairs['R_pair']
        # mass = (halo1['snr']+halo2['snr'])/2.
        ids = np.nonzero((length > bins_edges[ib-1]) * (length < bins_edges[ib]))[0]
        log.info('bin %d found n=%d ids' % (ib,len(ids)))
        list_prod_pdf , list_grid_pdf , prod2D_pdf ,  grid2D_kappa0 , grid2D_radius , n_pairs_used = get_prob_prod(ids)

        res_dict = {}
        res_dict['ib'] = ib
        res_dict['center'] = bins_centers[ib-1]
        res_dict['n_pairs_used'] = n_pairs_used
        res_dict['list_prod_pdf'] = list_prod_pdf
        res_dict['list_grid_pdf'] = list_grid_pdf
        res_dict['prod2D_pdf'] = prod2D_pdf
        res_dict['grid2D_kappa0'] = grid2D_kappa0
        res_dict['grid2D_radius'] = grid2D_radius

        list_res_dict.append(res_dict)

        pl.figure()
        pl.pcolormesh( grid2D_kappa0 , grid2D_radius, prod2D_pdf)

    filename_pickle = 'plotdata.length.pp2'
    tabletools.savePickle(filename_pickle,list_res_dict)

    pl.show()






def plotdata_vs_mass():

    # name_data = os.path.basename(config['filename_shears']).replace('.fits','')
    # filename_logpdf = 'logpdf.%s.pp2' % name_data
    # dict_logpdf = tabletools.loadPickle(filename_logpdf)
    # import pdb; pdb.set_trace()
    # logpdfs = dict_logpdf['logprob'] 
    # grids   = dict_logpdf['grid_centers']
    # list_ids = dict_logpdf['ids']

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    n_pairs = len(halo1)

    if 'cfhtlens' in filename_pairs:
        bins_snr_edges = [0,4,20]
        mass_param_name = 'snr'
    else:
        bins_snr_edges = [1e14,2e14,1e15]
        mass_param_name = 'm200'
    # bins_snr_centers = [ 3 , 6]
    bins_snr_centers = plotstools.get_bins_centers(bins_snr_edges)

    list_res_dict = []

    for ib in range(1,len(bins_snr_edges)):
        mass = (halo1[mass_param_name]+halo2[mass_param_name])/2.
        # mass = (halo1['snr']+halo2['snr'])/2.
        ids = np.nonzero((mass > bins_snr_edges[ib-1]) * (mass < bins_snr_edges[ib]))[0]
        log.info('bin %d: [%2.2e<mass<%2.2e], found n=%d ids' % (ib,  bins_snr_edges[ib-1], bins_snr_edges[ib], len(ids)))
        list_prod_pdf , list_grid_pdf , prod2D_pdf ,  grid2D_kappa0 , grid2D_radius , n_pairs_used = get_prob_prod(ids)

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


    filename_pickle = args.filename_config.replace('.yaml','.plotdata.mass.pp2')
    tabletools.savePickle(filename_pickle,list_res_dict)

    pl.show()



def plot_vs_mass():

    filename_pairs = config['filename_pairs']
    filename_halos = config['filename_pairs']
    filename_halos1 = filename_pairs.replace('.fits','.halos1.fits')
    filename_halos2 = filename_pairs.replace('.fits','.halos2.fits')

    halo1 = tabletools.loadTable(filename_halos1)
    halo2 = tabletools.loadTable(filename_halos2)
    n_pairs = len(halo1)

    filename_pickle = args.filename_config.replace('.yaml','.plotdata.mass.pp2')
    list_res_dict = tabletools.loadPickle(filename_pickle)
    nx=int(config['n_grid_2D']/2)

    mass_param_name = list_res_dict[0]['mass_param_name']
    mass = (halo1[mass_param_name]+halo2[mass_param_name])/2.

    pl.figure()
    pl.hist(mass)

    for ib,snr_bin in enumerate(list_res_dict):
        grid2D_kappa0 = snr_bin['grid2D_kappa0'][nx:,nx:]
        grid2D_radius = snr_bin['grid2D_radius'][nx:,nx:]       
        prod2D_pdf = snr_bin['prod2D_pdf'][nx:,nx:]
        prod2D_pdf,_,_,_ = mathstools.get_normalisation(np.log(prod2D_pdf))

        log.info('[%2.2e<mass<%2.2e]' % (snr_bin['bin_min'] , snr_bin['bin_max'] ))


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
        title_str= "mean halo %s \in [%2.2e , %2.2e] , n_pairs=%d'" % (mass_param_name , snr_bin['bin_min'] , snr_bin['bin_max'] , snr_bin['n_pairs_used'] )
        pl.title(title_str)
        filename_fig = 'figs/fig.mass.%02d.%s.png' % (ib,args.filename_config.replace('.yaml',''))
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



   
def main():


    valid_actions = ['test_kde_methods', 'plot_vs_mass', 'plotdata_vs_mass' , 'plot_vs_length', 'plotdata_vs_length']

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-n', '--n_results_files',type=int,  default=1, action='store', help='number of results files to use')
    parser.add_argument('-f', '--first_result_file',type=int, default=0, action='store', help='number of first result file to open')
    parser.add_argument('-p', '--save_plots', action='store_true', help='if to save plots')
    parser.add_argument('-a','--actions', nargs='+', action='store', help='which actions to run, available: %s' % str(valid_actions) )

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

    if 'test_kde_methods' in args.actions: test_kde_methods()
    if 'plot_vs_mass' in args.actions: plot_vs_mass()
    if 'plotdata_vs_mass' in args.actions: plotdata_vs_mass()
    if 'plot_vs_length' in args.actions: plot_vs_length()
    if 'plotdata_vs_length' in args.actions: plotdata_vs_length()
    if 'test' in args.actions: test()

    for ac in args.actions:
        if ac not in valid_actions:
            print 'invalid action %s' % ac


    

main()
