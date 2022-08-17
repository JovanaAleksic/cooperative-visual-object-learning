from __future__ import division
import numpy as np

def overlap_detection_3(treshold3, kappa, gamma, delta, perse, local_h):
    kappa_array = np.asarray(kappa)
    gamma_array = np.asarray(gamma)
    delta_array = np.asarray(delta)

    ratio = delta_array/gamma_array
    outlier_index = np.logical_and(gamma_array > local_h*treshold3, ratio > perse)
    sparse_region_index = np.logical_and(gamma_array > local_h*treshold3, ratio <= perse)
    dense_overlap_index = gamma_array <= local_h*treshold3
    outlier = sum(outlier_index)
    sparse_region = sum(sparse_region_index)
    dense_overlap = sum(dense_overlap_index)
    index_overlap_total = sparse_region_index + dense_overlap_index
    return outlier, sparse_region, dense_overlap, index_overlap_total
