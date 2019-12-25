# cython: language_level=3
# cython: infer_types=True
import warnings
from timeit import default_timer as timer

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

#from numpy import nanmean, nan, where, zeros, double, full, isnan

cdef double nan = np.nan


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
cpdef cy_avg_values_at_x_double(np.ndarray[double, ndim=1] uniq_x,
                                np.ndarray[double, ndim=2] leaf_ranges,
                                np.ndarray[double, ndim=1] leaf_slopes):
    """
    Compute the weighted average of leaf_slopes at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point.
    """
    cdef int nx, nslopes, i, j, n_nan, n_good
    cdef double xl, xr, slope, s, v
    cdef np.ndarray[double, ndim=2] slopes
    cdef np.ndarray[double, ndim=1] avg_value_at_x
    cdef np.ndarray[long, ndim=1] slope_counts_at_x


    nx = uniq_x.shape[0]
    nslopes = leaf_slopes.shape[0]
    slopes = np.zeros(shape=(nx, nslopes), dtype=np.double)
    cdef double[:,:] slopes_view = slopes
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range

    for j in range(nslopes):
        xl = leaf_ranges[j,0]
        xr = leaf_ranges[j,1]
        slope = leaf_slopes[j]
        # s = np.full(nx, slope)#, dtype=double)
        # s[np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]) )] = np.nan
        # slopes[:, i] = s

        # Compute slope all the way across uniq_x but then trim line so
        # slope is only valid in range xr; don't set slope on right edge
        for i in range(nx):
            if (uniq_x[i] < xl) or (uniq_x[i] >= xr):
                slopes_view[i, j] = nan
            else:
                slopes_view[i, j] = slope


    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    # Wrap nanmean() in catcher to avoid "Mean of empty slice" warning, which
    # comes from some rows being purely NaN; I should probably look at this sometime
    # to decide whether that's hiding a bug (can there ever be a nan for an x range)?
    # Oh right. We might have to ignore some leaves (those with single unique x values)

    # Compute:
    #   avg_value_at_x = np.mean(slopes[good], axis=1)  (numba doesn't allow axis arg)
    #   slope_counts_at_x = nslopes - np.isnan(slopes).sum(axis=1)

    # avg_value_at_x = np.nanmean(slopes, axis=1)
    # slope_counts_at_x = nslopes - np.isnan(slopes).sum(axis=1)

    avg_value_at_x = np.empty(shape=nx, dtype=np.double)
    cdef double[:] avg_value_at_x_view = avg_value_at_x
    slope_counts_at_x = np.empty(shape=nx, dtype=np.int)
    cdef long[:] slope_counts_at_x_view = slope_counts_at_x
    for i in range(nx):
        n_good = 0
        s = 0.0
        for j in range(nslopes):
            v = slopes[i,j]
            if v!=nan:
                s += v
                n_good += 1
        if n_good>0:
            avg_value_at_x_view[i] = s / n_good
        else:
            avg_value_at_x_view[i] = nan

        slope_counts_at_x_view[i] = n_good

    #
    # avg_value_at_x = np.zeros(shape=nx)
    # slope_counts_at_x = np.zeros(shape=nx)
    # for i in range(nx):
    #     n_nan = 0
    #     for j in range(nslopes):
    #         if
    #     n_nan = sum(np.isnan(slopes[i, :]))
    #     avg_value_at_x[i] = nan if n_nan==nslopes else np.nanmean(slopes[i, :])
    #     slope_counts_at_x[i] = nslopes - n_nan

    # return average slope at each unique x value and how many slopes included in avg at each x
    return avg_value_at_x, slope_counts_at_x


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
cpdef cy_avg_values_at_x_long(np.ndarray[long, ndim=1] uniq_x,
                              np.ndarray[long, ndim=2] leaf_ranges,
                              np.ndarray[long, ndim=1] leaf_slopes):
    """
    Compute the weighted average of leaf_slopes at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point.
    """
    cdef int nx, nslopes, i
    cdef long xl, xr
    cdef double slope
    cdef np.ndarray[double, ndim=2] slopes, avg_value_at_x, slope_counts_at_x

    nx = len(uniq_x)
    nslopes = len(leaf_slopes)
    slopes = np.zeros(shape=(nx, nslopes))
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range

    for i in range(nslopes):
        xl = leaf_ranges[i,0]
        xr = leaf_ranges[i,1]
        slope = leaf_slopes[i]

        # s = np.full(nx, slope)#, dtype=double)
        # s[np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]) )] = np.nan
        # slopes[:, i] = s

        # Compute slope all the way across uniq_x but then trim line so
        # slope is only valid in range xr; don't set slope on right edge
        slopes[:, i] = np.where( (uniq_x < xl) | (uniq_x >= xr), nan, slope)


    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    # Wrap nanmean() in catcher to avoid "Mean of empty slice" warning, which
    # comes from some rows being purely NaN; I should probably look at this sometime
    # to decide whether that's hiding a bug (can there ever be a nan for an x range)?
    # Oh right. We might have to ignore some leaves (those with single unique x values)

    # Compute:
    #   avg_value_at_x = np.mean(slopes[good], axis=1)  (numba doesn't allow axis arg)
    #   slope_counts_at_x = nslopes - np.isnan(slopes).sum(axis=1)
    avg_value_at_x = np.zeros(shape=nx)
    slope_counts_at_x = np.zeros(shape=nx)
    for i in range(nx):
        row = slopes[i, :]
        n_nan = sum(np.isnan(row))
        avg_value_at_x[i] = nan if n_nan==nslopes else np.nanmean(row)
        slope_counts_at_x[i] = nslopes - n_nan

    # return average slope at each unique x value and how many slopes included in avg at each x
    return avg_value_at_x, slope_counts_at_x
