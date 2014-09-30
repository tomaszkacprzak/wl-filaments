import pyfits
import numpy as np
import pylab as pl
import scipy.interpolate as interp
import cosmology
import tabletools
import yaml, argparse, sys, logging 
from sklearn.neighbors import BallTree as BallTree
import galsim


filename_big = 'big_halos.fits'
halocat = tabletools.loadTable(filename_big,table_name='big')
sorting = np.argsort(halocat['M200'])
biggest_halo = halocat[sorting[-3]]

biggest_halo['RA'],biggest_halo['DEC'] = 0. , 0.

dtheta=0.1
lenscat={}
lenscat['ra'] = np.random.uniform(biggest_halo['RA']-dtheta,biggest_halo['RA']+dtheta,1000)
lenscat['dec'] = np.random.uniform((biggest_halo['DEC']-dtheta),(biggest_halo['DEC']+dtheta),1000)
lenscat['z'] = lenscat['dec']*0 + biggest_halo['Z'] * 2.

conc=biggest_halo['RVIR']/(biggest_halo['RS']/1e3)


halo1_ra_arcsec, halo1_de_arcsec = cosmology.deg_to_arcsec(biggest_halo['RA'], biggest_halo['DEC']) 
shear_ra_arcsec, shear_de_arcsec = cosmology.deg_to_arcsec(lenscat['ra'], lenscat['dec']) 
nfw1=galsim.NFWHalo(conc=conc,redshift=biggest_halo['Z'],mass=biggest_halo['M200'],omega_m = cosmology.cospars.omega_m,halo_pos=galsim.PositionD(x=halo1_ra_arcsec,y=halo1_de_arcsec))
(g1,g2,_)=nfw1.getLensing(pos=(shear_ra_arcsec, shear_de_arcsec),z_s=lenscat['z'])


# conc=biggest_halo['RVIR']/(biggest_halo['RS']/1e3)
# nfw1=galsim.NFWHalo(conc=conc,redshift=biggest_halo['Z'],mass=biggest_halo['M200'],omega_m = cosmology.cospars.omega_m,halo_pos=galsim.PositionD(x=biggest_halo['RA']*3600,y=biggest_halo['DEC']*3600))

import filaments_tools
filaments_tools.plot_pair(biggest_halo['RA'],biggest_halo['DEC'],biggest_halo['RA'],biggest_halo['DEC'],lenscat['ra'],lenscat['dec'],g1,g2)

rotation_angle = 30 * np.pi / 180.

pair_ra_dec,pair_de_dec = biggest_halo['RA'] , biggest_halo['DEC']


halo1_u_rot , halo1_v_rot = filaments_tools.rotate_vector(rotation_angle, biggest_halo['RA'], biggest_halo['DEC'])
shear_u_rot , shear_v_rot = filaments_tools.rotate_vector(rotation_angle, lenscat['ra'], lenscat['dec'])
shear_g1_rot , shear_g2_rot = filaments_tools.rotate_shear(rotation_angle, lenscat['ra'], lenscat['dec'], g1, g2)
filaments_tools.plot_pair(halo1_u_rot, halo1_v_rot, halo1_u_rot, halo1_v_rot, shear_u_rot, shear_v_rot, shear_g1_rot, shear_g2_rot)


# -------

pair_ra_deg , pair_de_deg = biggest_halo['RA']+dtheta , biggest_halo['DEC']+dtheta
halo1_u , halo1_v  = filaments_tools.linearise_coords(biggest_halo['RA'], biggest_halo['DEC'], pair_ra_dec, pair_de_dec,local=True)
shear_u , shear_v  = filaments_tools.linearise_coords(lenscat['ra'], lenscat['dec'], pair_ra_dec, pair_de_dec,local=True)

shear_u_rot , shear_v_rot = filaments_tools.rotate_vector(rotation_angle, shear_u, shear_v)
halo1_u_rot , halo1_v_rot = filaments_tools.rotate_vector(rotation_angle, halo1_u, halo1_v)
shear_g1_rot , shear_g2_rot = filaments_tools.rotate_shear(rotation_angle, shear_u, shear_v, g1, g2)

filaments_tools.plot_pair(halo1_u, halo1_v, halo1_u, halo1_v, shear_u, shear_v, g1, g2)
filaments_tools.plot_pair(halo1_u_rot, halo1_v_rot, halo1_u_rot, halo1_v_rot, shear_u_rot, shear_v_rot, shear_g1_rot, shear_g2_rot)
pl.show()