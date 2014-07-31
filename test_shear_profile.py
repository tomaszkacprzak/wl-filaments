import numpy as np; import pylab as pl;
rc=0.5
sig=0.25
x=np.linspace(-10,10,1000)
l=8
y=1./(1+np.exp( (np.abs(x)-l) /sig) + (0/rc)**2)

pl.plot(x,y); 
pl.axvline(-l)
pl.axvline(l)

pl.show()
