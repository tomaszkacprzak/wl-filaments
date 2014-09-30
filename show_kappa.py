import pyfits
import numpy
import pylab
import scipy.interpolate as interp

DEG2ARCMIN = 1./60**2

def intersect_rect(rect1,rect2):

    min1_out = max(rect1[0],rect2[0])
    max1_out = min(rect1[1],rect2[1])
    min2_out = max(rect1[2],rect2[2])
    max2_out = min(rect1[3],rect2[3])

    return [min1_out, max1_out, min2_out, max2_out]

def select_rect(x,rect):

    select1 = x['ra'] > rect[0]
    select2 = x['ra'] < rect[1]
    select3 = x['dec'] > rect[2]
    select4 = x['dec'] < rect[3]

    select  = select1 * select2 * select3 * select4


    return x[select]


def rect_area(rect):

    return abs(rect[1] - rect[0]) * abs(rect[3] - rect[2])


filename_lenscat = 'aardvarkv1.0_des_lenscat_s2n20.86.fit'
lenscat = pyfits.getdata(filename_lenscat)
ra = lenscat['ra']
dec = lenscat['dec']
kappa = lenscat['kappa']

# show the cover of the catalog


# get the rect of lenscat
# lenscat_rect = [min(ra) , max(ra) , min(dec) , max(dec)]
lenscat_rect = [344.5 , 348. , min(dec) , max(dec)]
lenscat_area = rect_area(lenscat_rect)
lenscat = select_rect(lenscat,lenscat_rect)
print 'lenscat_rect',lenscat_rect
print 'n_gals=%d, area=%f, density=%f' % (len(lenscat),lenscat_area,float(len(lenscat))/lenscat_area*DEG2ARCMIN)

filename_halos = 'Aardvark_v1.0_halos_r1_rotated.0.fit'
halocat = pyfits.getdata(filename_halos)

ra = halocat['ra']
dec = halocat['dec']

halocat_rect = [min(ra) , max(ra) , min(dec) , max(dec)]
halocat_area = rect_area(halocat_rect)
print 'halocat_rect', halocat_rect
print 'n_halos=%d, area=%f, density=%f' % (len(halocat),halocat_area,float(len(halocat))/halocat_area)

common_rect = intersect_rect(lenscat_rect,halocat_rect)
print 'common_rect' , common_rect

halocat_common = select_rect(halocat,common_rect)
halocat_area = rect_area(common_rect)
print 'n_halos=%d, area=%f, density=%f' % (len(halocat_common),halocat_area,float(len(halocat_common))/halocat_area)

perm=numpy.random.permutation(len(lenscat))
plenscat = lenscat[perm]
pylab.scatter(plenscat[:10000]['ra'],plenscat[:10000]['dec'],'x')
pylab.scatter(halocat_common['ra'],halocat_common['dec'],'o')
pylab.show()

#  print some stats of the halo catalog

pylab.scatter(halocat_common['R200'],halocat_common['NGALS'])
filename_fig = 'figs/clusters_r200_vs_ngals.png'
pylab.savefig(filename_fig)
pylab.close()
print 'saved' , filename_fig

# make cutouts of the kappa around massive clusters

sort_ngals = numpy.argsort(halocat_common['NGALS'])
n_clusters_to_show = 3
window_size = 1 # deg

map_nx = 1000
map_ny = 1000

for i in range(n_clusters_to_show):
    cluster_id = sort_ngals[i]
    this_cluster = halocat_common[cluster_id]
    xmin = this_cluster['ra'] - window_size
    xmax = this_cluster['ra'] + window_size
    ymin = this_cluster['dec'] - window_size
    ymax = this_cluster['dec'] + window_size
    this_window_rect = [xmin,xmax,ymin,ymax]
    this_lenscat = select_rect(lenscat,this_window_rect)
    print 'cluster %d selected %d galaxies' % (i,len(this_lenscat))

    if len(this_lenscat) == 0:
        print 'cluster outside lenscat, skipping'
        continue

    xi = numpy.linspace(xmin, xmax, map_nx)
    yi = numpy.linspace(ymin, ymax, map_ny)
    xi, yi = numpy.meshgrid(xi, yi)

    tra = numpy.array(this_lenscat['ra'],ndmin=2)
    tde = numpy.array(this_lenscat['dec'],ndmin=2)
    griddata_points = numpy.concatenate((tra,tde),axis=0).T
    griddata_values = this_lenscat['KAPPA']

    print 'running interp'
    zi = interp.griddata(griddata_points,griddata_values,(xi,yi))

    pylab.figure()
    pylab.clf()
    pylab.imshow(zi,interpolation='nearest',extent=(xmin,xmax,ymin,ymax))
    pylab.colorbar()
    pylab.scatter(this_cluster['ra'],this_cluster['dec'])
    pylab.show()






    

