import cosmology
import galsim

omega_m = 0.3
omega_lam = 0.7
z_gal = 2.
z_lens = 0.5
nfw = galsim.NFWHalo(mass=1e14,conc=0.7,omega_m=omega_m,omega_lam=omega_lam,redshift=z_lens)
print nfw._NFWHalo__ks(z_s=z_gal)

cosmology.cospars.omega_m = omega_m
cosmology.cospars.omega_lambda = omega_lam
print cosmology.get_sigma_crit(z_gal=z_gal,z_lens=z_lens)
