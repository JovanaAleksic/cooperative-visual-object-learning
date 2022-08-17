from __future__ import division
import numpy as np

def confidence_computation(outlier, sparse_region, dense_overlap):
    # outlier_measure = np.sum(outlier)
    # sparse_region_measure = np.sum(sparse_region)
    # edge_region_measure = np.sum(edge_region)
    # dense_overlap = np.sum(dense_overlap)
    # locally_dense_overlap = np.sum(locally_dense_overlap)

    # confidence_outlier = sum(outlier_measure > (lenen*treshold1))
    # confidence_dense_overlap = sum(dense_overlap > (lenen*treshold1))
    # confidence_local_overlap = sum (locally_dense_overlap > (lenen*treshold1))  #mechanism is wrong!
    # if confidence_outlier > confidence_dense_overlap:
    #     confidence = 1
    #     new = 1
    # elif confidence_local_overlap > confidence_dense_overlap:
    #     confidence = 1
    #     new = 0
    # else:
    #     confidence = 0
    #     new = 0
    # return confidence, new

    measure = outlier/(outlier + sparse_region + dense_overlap)
    if measure >= 0.5:
        confidence = (measure-0.5)*2
        new = 1
    else:
        confidence = (0.5-measure)*2
        new = 0

    return confidence, new, measure
