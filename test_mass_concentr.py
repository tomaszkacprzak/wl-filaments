import numpy as np
import pylab as pl
import nfw
halo = nfw.NfwHalo()

halo.z_cluster = 0.

m_range = np.logspace(10,15.5)
c_duffy = []
c_dutton = []

for m200 in m_range:
	halo.M_200 = m200
	c_duffy.append(halo.get_concentr(method="Duffy"))
	c_dutton.append(halo.get_concentr(method="Dutton"))

# reproduce fig 7 Dutton
pl.plot(np.log10(m_range),np.log10(c_duffy),label="Duffy+ 2008",c='r')
pl.plot(np.log10(m_range),np.log10(c_dutton),label="Dutton+ 2014",c='g')
pl.legend()
pl.xlabel('log10 M_200')
pl.ylabel('log10 c_200')
# pl.xscale('log')
# pl.yscale('log')
pl.show()
