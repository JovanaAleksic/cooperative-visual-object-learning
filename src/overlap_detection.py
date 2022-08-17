from __future__ import division
import numpy as np

# def overlap_detection(treshold, kappa, gamma, delta):
#     kappa_array = np.asarray(kappa)
#     gamma_array = np.asarray(gamma)
#     delta_array = np.asarray(delta)
#
#     kappa_scaled = kappa_array / kappa_array.max()
#     gamma_scaled = gamma_array / gamma_array.max()
#     delta_scaled = delta_array / delta_array.max()
#
#     index_kappa = kappa_scaled < treshold  #different treshold should be used
#     index_gamma = gamma_scaled < treshold
#     index_delta = delta_scaled < treshold
#
#     sum = index_kappa+5*index_gamma+10*index_delta
#
#     outlier = sum < 2
#     sparse_region = np.logical_and(sum > 10, sum < 15)
#     edge_region = np.logical_and(sum > 2, sum < 10)
#     dense_overlap = sum > 15
#     locally_dense_overlap = np.logical_and(sum < 16, sum > 14)
#     return outlier, sparse_region, edge_region, dense_overlap, locally_dense_overlap

def overlap_detection(treshold1, treshold2, kappa, gamma, delta):
    kappa_array = np.asarray(kappa)
    gamma_array = np.asarray(gamma)
    delta_array = np.asarray(delta)

    index = kappa_array/gamma_array/delta_array
    outlier_index = index < treshold1
    sparse_region_index = np.logical_and(index >= treshold1, index < treshold2)
    dense_overlap_index = index >= treshold2
    outlier = sum(outlier_index)
    sparse_region = sum(sparse_region_index)
    dense_overlap = sum(dense_overlap_index)
    index_overlap_total = sparse_region_index + dense_overlap_index
    return outlier, sparse_region, dense_overlap, index_overlap_total
