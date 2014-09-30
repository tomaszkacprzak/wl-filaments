import numpy as np; import pylab as pl;

print 'testing'
rc=0.2
sig=0.
x=np.linspace(-10,10,1000)
R_start=1
l=10/2.
length = l-R_start
y=1./(1+np.exp( (np.abs(x)-length) /sig) + (0/rc)**2)

pl.plot(x,y); 
pl.axvline(-l+1)
pl.axvline(-l)
pl.axvline(l-1)
pl.axvline(l)

print 'plotting'
pl.show()
