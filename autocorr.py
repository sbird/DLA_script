"""Module to compute the autocorrelation function of a field
"""

import numpy as np

def autocorr_python(field):
    """Pure python implementation of the 1-d autocorrelation function.
       This will be very slow, and is also better done with np.correlate.
       Assume it has been normalised elsewhere
       Arguments:
          field - field to find the autocorrelation of
    """
    #For a 2d square field this will be (N, N), and N^D = M
    dims = np.shape(field)
    #To store output - spacing evenly in grid units
    autocorr = np.zeros(dims[0])
    count = np.zeros(dims[0])
    #Do correlation of each cell with each other cell in turn: O(M^2)
    for a in xrange(0,dims[0]):
        for b in xrange(0,dims[0]):
            #Calculate bin: distance between the two points
            r = b - a
            autocorr[r] += field[a]*field[b]
            count[r] += 1
    #Normalise wrt number of cells
    for r in xrange(0, dims[0]):
        autocorr[r]/=count[r]

    return autocorr

def autocorr_list(plist, nbins, size, weight=1,norm=True):
    """Find the autocorrelation function from a sparse list of discrete tracer points.
       The field is assumed to be 1 at these points and zero elsewhere
       list - list of points to autocorrelate. A tuple length n of 1xP arrays:
       the output of an np.where on an n-d field
       nbins - number of bins in output autocorrelation function
       size - size of the original field (assumed square), so field has dimensions (size,size..) n times
       weight - weight each point has: use 1/(avg. density)
       norm - If true, normalise by the number of possible cells in each bin
    """
    #Make an array of shape (n,P)
    plist = np.array(plist)
    (dims, points) = np.shape(plist)
    #Bin autocorrelation, must cover sqrt(dims)*size
    #so each bin has size sqrt(dims)*size /nbins
    autocorr = np.zeros(nbins)
    count = np.zeros(nbins)
    for b in xrange(points):
        for a in xrange(points):
            rr = distance(plist[:,a], plist[:,b])
            #Which bin to add this one to?
            cbin = np.floor(rr * nbins / size*np.sqrt(dims))
            autocorr[cbin]+=weight

    if norm:
        for nn in xrange(0,nbins):
            #Count number of square bins in a circle of radius sqrt(dims)*size
            #This is 4 * (quarter circle)
            # = 4 * sum(y < r) \sqrt(r^2-y^2)
            #Maximal radius in this bin
            rr = (1+nn)*np.sqrt(dims)*size/(1.*nbins)
            #Vector of y values
            yy = np.arange(0,np.floor(rr))
            #Vector of integrands along x axis
            count[nn] = 4*np.sum(np.ceil(np.sqrt(rr**2 - yy**2)))
        #Take off the modes in previous bin to get an annulus
        for nn in xrange(nbins-1,0,-1):
            count[nn] -= count[nn-1]
        for nn in xrange(0,nbins):
            autocorr[nn]/=count[nn]
    return autocorr


def distance(a, b):
    """Compute the absolute distance between two points"""
    return np.sqrt(np.sum((a-b)**2))
