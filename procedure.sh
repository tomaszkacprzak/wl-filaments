# measure BOSS galaxies in CFHTLens
python ~/code/wl-filaments/halos_cfhtlens.py -c test.yaml -f 0 -n 10 -a select_lrgs > log.halos_cfhtlens.select_lrgs.txt

# measure halos and stack
python ~/code/wl-filaments/halos_cfhtlens.py -c test.yaml -f 0 -n 100 -a fit_halos > log.halos_cfhtlens.fit_halos.txt

# select filaments
python ~/code/wl-filaments/filaments_cfhtlens.py -c test.yaml -a get_pairs
python ~/code/wl-filaments/filaments_cfhtlens.py -c test.yaml -a get_stamps

# make clone - with kappa0=0 
python ~/code/wl-filaments/filaments_selffit.py -c lrgs.yaml -a get_clone

# make random point
python ~/code/wl-filaments/halos_cfhtlens.py -c lrgs.yaml -a randomise_halos2
python ~/code/wl-filaments/filaments_cfhtlens.py -c random.yaml -a get_random_pairs
python ~/code/wl-filaments/filaments_cfhtlens.py -c random.yaml -a get_stamps
python ~/code/wl-filaments/filaments_cfhtlens.py -c random.yaml -a add_nfw

# fit all
python ~/code/wl-filaments/filaments_fit.py -c lrgs.yaml 
python ~/code/wl-filaments/filaments_fit.py -c random.yaml 
python ~/code/wl-filaments/filaments_fit.py -c clone.yaml 




