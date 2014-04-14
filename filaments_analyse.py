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


def process_results():

    name_data = os.path.basename(config['filename_shears']).replace('.fits','')
    # filename_results_cat = 'results.stats.%s.cat' % name_data
    # stats = tabletools.loadTable(filename_results_cat,dtype=filaments_model_2hf.dtype_stats)

    n_per_file = 10
    n_files = args.n_results_files
    log.info('n_files=%d',n_files)

    n_params = 4
    n_colors = 10
    ic =0 
    n_grid = 200
    list_logprob = [None]*n_params
    list_grid_centers = [None]*n_params
    list_ids = []

    n_missing=0
    ia=0

    for ip in range(n_params):
        list_logprob[ip] = []
        list_grid_centers[ip] = []


    for nf in range(n_files):
    # for nf in range(2):

        filename_pickle = 'results/results.chain.%04d.%04d.%s.pp2'  % (nf*n_per_file, (nf+1)*n_per_file , name_data)
        try:
            results_pickle = tabletools.loadPickle(filename_pickle)
            log.info('%4d %s' , nf , filename_pickle)
        except:
            log.info('missing %s' % filename_pickle)
            n_missing +=1
            ia+=1
            continue

        for ni in range(n_per_file):

            for ip in range(n_params):
                prob = results_pickle[ni]['list_prob_marg'][ip]
                nans = np.nonzero(np.isnan(prob))[0]
                if len(nans) > 0:
                    log.info('%d %s param=%d n_nans=%d', ni, filename_pickle , ip, len(np.nonzero(np.isnan(prob))[0]) )
                prob /= np.sum(prob)
                logprob = np.log(prob)
                nans = np.nonzero(np.isnan( logprob))[0]
                infs = np.nonzero(np.isinf( logprob))[0]
                zeros = np.nonzero( logprob == 0.0)[0]
                
                list_logprob[ip].append(logprob)
                list_ids.append(ia)
            ia+=1

    log.info('n_missing=%d' , n_missing)
    for ip in range(n_params):  
        list_logprob[ip] = np.array(list_logprob[ip])
        list_grid_centers[ip] = np.array(results_pickle[0]['list_params_marg'][ip])

    filename_DeltaSigma = 'logpdf.%s.pp2' % name_data
    tabletools.savePickle(filename_DeltaSigma, { 'grid_centers' : list_grid_centers , 'ids' : list_ids ,  'logprob' : list_logprob } )    



    


    
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



def plot_prob_product():

    name_data = os.path.basename(config['filename_shears']).replace('.fits','')
    filename_logpdf = 'logpdf.%s.pp2' % name_data
    dict_logpdf = tabletools.loadPickle(filename_logpdf)
    logpdfs = dict_logpdf['logprob'] 
    grids   = dict_logpdf['grid_centers']

    # 'normalising'
    # for il in range(logpdf_DeltaSigma.shape[0]):
    #     logpdf_DeltaSigma[il,:] -= np.log(np.sum(np.exp(logpdf_DeltaSigma[il,:])))

    n_params = 4
    n_step = 1000
    n_pairs = logpdfs[0].shape[0]
    n_colors = n_pairs/n_step+1
    colors = plotstools.get_colorscale(n_colors)

    list_conf = [None]*n_params

    n_use = 0
    n_nan_pairs = 0
    param_names = {0:'kappa0',1:'radius',2:'h1M200',3:'h2M200'}

    pl.figure()
    for ip in range(n_params):
        pl.subplot(2,2,ip)

        ic=0
        list_conf[ip] = []
        logpdf = logpdfs[ip]
        grid_pdf = grids[ip]
        perm=np.random.permutation(n_pairs)
        logpdf = logpdf[perm]
        param_name = param_names[ip]


        for ia in range(n_pairs):


            nans=np.nonzero(np.isnan(logpdf[ia]))[0]
            if len(nans)>0:
                
                n_nan_pairs +=1
                # print 'min(nans) , max(nans) , len(nans) , n_nan_pairs/ia' , min(nans) , max(nans) , len(nans) , n_nan_pairs , ia, n_nan_pairs/float(ia)
                # if np.isnan(logpdf_DeltaSigma[ia,0]):
                    # logpdf_DeltaSigma[il,:] -= np.log(np.sum(np.exp(logpdf_DeltaSigma[il,:])))
                # logpdf_DeltaSigma[ia,min(nans):]=logpdf_DeltaSigma[ia,min(nans)-1]

                mask = np.isnan(logpdf[ia,:])
                logpdf[ia,:] = np.exp(logpdf[ia,:])
                logpdf[ia,mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), logpdf[ia,~mask]) / 2.
                logpdf[ia,:] = np.log(logpdf[ia,:]) 

            if (ia+1) % n_step == 0:
                print 'passing' , ia+1

                # sum_log_DeltaSigma = np.sum(logpdf_DeltaSigma[:ia],axis=0)          
                sum_logpdf = np.sum(logpdf[:ia,:],axis=0)
                prod_pdf , prod_log_pdf , _ , _ = mathstools.get_normalisation(sum_logpdf)  
                max_pdf , pdf_err_hi , pdf_err_lo = mathstools.estimate_confidence_interval(grid_pdf , prod_pdf)
                list_conf[ip].append([ia,max_pdf , pdf_err_hi , pdf_err_lo])
            
                pl.plot(grid_pdf , prod_pdf , '-x', label='n_pairs=%d max=%5.4f +/- %5.4f/%5.4f' % (ia, max_pdf , pdf_err_hi , pdf_err_lo) , color=colors[ic])
                ic+=1

        true_params=[0.0,0,14,14]
        pl.legend()
        pl.axvline( true_params[ip] )
        pl.xlim( [ config[param_name]['box']['min'] , config[param_name]['box']['max'] ])
        # pl.xlabel(r'$\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}$')
        pl.xlabel(param_name)
        pl.ylabel('likelihood')

        list_conf[ip] = np.array(list_conf[ip])

    pl.show()
        
        # pl.figure()
        # pl.plot(list_conf[ip][:,0],list_conf[ip][:,2])
        # pl.plot(list_conf[ip][:,0],list_conf[ip][:,3])
        # print list_conf

    # filename_fig = 'figs/prod_DeltaSigma.%s.png' % name_data
    # pl.savefig(filename_fig)
    # log.info( 'saved %s' , filename_fig )

  




   
def main():

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    parser.add_argument('-c', '--filename_config', type=str, default='filaments_config.yaml' , action='store', help='filename of file containing config')
    parser.add_argument('-n', '--n_results_files',type=int, action='store', help='number of results files to use')
    parser.add_argument('-p', '--save_plots', action='store_true', help='if to save plots')
    parser.add_argument('-fg', '--filename_shears', type=str, default='shears_bcc_g.fits' , action='store', help='filename of file containing shears in binned format')
    parser.add_argument('-a','--actions', nargs='+', default=['plot_prob_product'] , action='store', help='which actions to run, available: {process_results, plot_prob_product, test_kde_methods} ' )

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

    if args.actions==[]:
        'choose one or more actions: '

    if 'process_results' in args.actions: process_results()
    if 'plot_prob_product' in args.actions: plot_prob_product()
    if 'test_kde_methods' in args.actions: test_kde_methods()

main()
