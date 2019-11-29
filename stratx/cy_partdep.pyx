# cython: language_level=3
import warnings
from timeit import default_timer as timer

cimport numpy as np

np.import_array()
import numpy

from numpy import nanmean, nan, where, zeros, double, full

cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
cpdef np.ndarray[float, ndim=1] cy_avg_values_at_x(uniq_x:np.ndarray[:],
                                                   leaf_ranges:np.ndarray[:,:],
                                                   leaf_slopes:np.ndarray[double]
                                                   , verbose:bool):
    """
    Compute the weighted average of leaf_values at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point.
    
    for xr, slope in zip(leaf_ranges, leaf_slopes):
        s = np.full(nx, slope, dtype=float)
        s[np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]) )] = np.nan
        slopes[:, i] = s
        i += 1

    """
    cdef float start = timer()
    cdef int nx = len(uniq_x)
    cdef int nslopes = len(leaf_slopes)
    cdef np.ndarray[double, ndim=2] slopes = zeros(shape=(nx, nslopes))
    cdef int i = 0  # unique x value (column in slopes matrix) index; we get a slope line for each range x_i to x_i+1
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range
#    for xr, slope in zip(leaf_ranges, leaf_slopes):
    cdef np.ndarray[double] s = full(nx, fill_value=nan, dtype=double)

    for xr, slope in zip(leaf_ranges, leaf_slopes):
        s.fill(slope)
        s[where( (uniq_x < xr[0]) | (uniq_x >= xr[1]) )] = nan
        slopes[:, i] = s
        i += 1

    """
    for j in range(nslopes):
        for i in range(nx):
            if (uniq_x[i] < leaf_ranges[j,0]) or (uniq_x[i] >= leaf_ranges[j,1]):
                slopes[i,j] = leaf_slopes[j]
        # s.fill(leaf_slopes[r])
        # now trim line so it's only valid in range xr;
        # don't set slope on right edge
        # s[where( (uniq_x < leaf_ranges[r,0]) | (uniq_x >= leaf_ranges[r,1]) )] = nan
        # slopes[:, i] = s
        # i += 1
    """
    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    # Wrap nanmean() in catcher to avoid "Mean of empty slice" warning, which
    # comes from some rows being purely NaN; I should probably look at this sometime
    # to decide whether that's hiding a bug (can there ever be a nan for an x range)?
    # Oh right. We might have to ignore some leaves (those with single unique x values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_value_at_x = nanmean(slopes, axis=1)

    stop = timer()
    if verbose: print(f"avg_value_at_x {stop - start:.3f}s")
    return avg_value_at_x # return average slope at each unique x value
