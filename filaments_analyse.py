import matplotlib as mpl
mpl.use('tkagg')
import os, yaml, argparse, sys, logging , pyfits, galsim, emcee, tabletools, cosmology, filaments_tools, plotstools, mathstools, scipy, scipy.stats
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

    name_data = 'shears_bcc_e'
    filename_results_cat = 'results.stats.%s.cat' % name_data
    stats = tabletools.loadTable(filename_results_cat,dtype=filaments_model_2hf.dtype_stats)

    n_files = 300
    n_per_file = 10

    from scipy.stats import kde

    list_DeltaSigma = []

    n_colors = 10
    ic =0 
    colors = plotstools.get_colorscale(n_colors)

    ia=1
    for nf in range(n_files):
    # for nf in range(10):

        filename_pickle = 'results/results.chain.%04d.%04d.%s.pp2'  % (nf*n_per_file, (nf+1)*n_per_file , name_data)
        try:
            results_pickle = tabletools.loadPickle(filename_pickle)
        except:
            print 'missing %s' % results_pickle
            continue

        for ni in range(n_per_file):
            
            k = kde.gaussian_kde(results_pickle[ni]['flatchain'][0][:,0])
            grid_DeltaSigma = np.linspace(0,0.2,200)
            prob_DeltaSigma = k(grid_DeltaSigma)
            logprob_DeltaSigma = np.log(prob_DeltaSigma)
            list_DeltaSigma.append(logprob_DeltaSigma)
            

            print filename_pickle , ni , grid_DeltaSigma[prob_DeltaSigma.argmax()]
            if ia % 300 == 0:

                arr_list_DeltaSigma = np.array(list_DeltaSigma)
                sum_log_DeltaSigma = np.sum(arr_list_DeltaSigma,axis=0)
                prod_DeltaSigma , prod_log_DeltaSigma , _ , _ = mathstools.get_normalisation(sum_log_DeltaSigma)

                pl.plot(grid_DeltaSigma , prod_DeltaSigma , label='using %d pairs' % ia , color=colors[ic])
                ic+=1

            ia += 1

    pl.legend()
    pl.xlabel(r'$\Delta \Sigma 10^{14} * M_{*} \mathrm{Mpc}^{-2}$')
    pl.ylabel('likelihood')
    filename_fig = 'figs/prod_DeltaSigma.shears_bcc_e.png'
    pl.savefig(filename_fig)
    log.info( 'saved %s' , filename_fig )
    pl.show()

  




   
def main():

    description = 'filaments_fit'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3 ), help='integer verbosity level: min=0, max=3 [default=2]')
    # parser.add_argument('-o', '--filename_output', default='test2.cat',type=str, action='store', help='name of the output catalog')
    # parser.add_argument('-c', '--filename_config', default='test2.yaml',type=str, action='store', help='name of the yaml config file')
    parser.add_argument('-f', '--first', default=-1,type=int, action='store', help='first pair to process')
    parser.add_argument('-n', '--num', default=-1,type=int, action='store', help='number of pairs to process')
    parser.add_argument('-p', '--save_plots', action='store_true', help='if to save plots')
    parser.add_argument('-fg', '--filename_shears', type=str, default='shears_bcc_g.fits' , action='store', help='filename of file containing shears in binned format')
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

    filaments_model_1f.log = log
    # plotstools.log = log

    # grid_search(pair_info,shears_info)
    # test_model(shears_info,pair_info)

    # fit_single_halo()
    # fit_single_filament(save_plots=args.save_plots)
    # process_results()
    # analyse_stats_samples()
    process_results()


main()