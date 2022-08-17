from __future__ import division
import numpy as np

def overlap_detection_2(treshold1, treshold2, kappa, gamma, delta, local_h, local_g):

    kappa_array = np.asarray(kappa)
    gamma_array = np.asarray(gamma)
    delta_array = np.asarray(delta)

    outlier_index = np.logical_and(gamma_array > local_h*treshold1, delta_array > local_g*treshold2)
    sparse_region_index = np.logical_and(gamma_array > local_h*treshold1, delta_array <= local_g*treshold2)
    dense_overlap_index = gamma_array <= local_h*treshold1
    outlier = sum(outlier_index)
    sparse_region = sum(sparse_region_index)
    dense_overlap = sum(dense_overlap_index)
    index_overlap_total = sparse_region_index + dense_overlap_index
    return outlier, sparse_region, dense_overlap, index_overlap_total
