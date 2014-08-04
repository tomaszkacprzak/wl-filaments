import numpy as np; import pylab as pl;
rc=0.2
sig=0.2
x=np.linspace(-10,10,1000)
R_start=2
l=8
y=1./(1+np.exp( (np.abs(x)-l+R_start) /sig) + (0/rc)**2)

pl.plot(x,y); 
pl.axvline(-l)
pl.axvline(l)

pl.show()
