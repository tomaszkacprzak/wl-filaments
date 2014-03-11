import plotstools , logging;
plotstools.log.setLevel(logging.DEBUG)
from numpy import *
import pylab as pl

x=linspace(-20,20,10000)
y=exp(-x**2./2.)/sqrt(2*pi)
y /= sum(y)

pl.plot(x,y)
level , _ = plotstools.get_sigma_contours_levels(y,[1])
print level

diff = abs(y-level)
ix1,ix2 = diff.argsort()[:2]
x1 , x2 = x[ix1] , x[ix2]

pl.plot(x,level*ones_like(x))
pl.axvline(x=x1,linewidth=1, color='r')
pl.axvline(x=x2,linewidth=1, color='r')


pl.show()

