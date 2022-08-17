from __future__ import division
import numpy as np
from overlap_measures1 import overlap_measures1


def different_views(training_data, test_data, n_neighbors, neigh, treshold1, treshold2, treshold3):

    if len(training_data) < 5:
        reward = 1.0
    else:
        kappa1, gamma1, delta1, distances, indices = overlap_measures1(neigh, n_neighbors, training_data, test_data)
        # outlier, sparse_region, dense_overlap, index_total = \
        #     overlap_detection(treshold1, treshold2, kappa1, gamma1, delta1)

        kappa_array = np.asarray(kappa1)
        gamma_array = np.asarray(gamma1)
        delta_array = np.asarray(delta1)

        index = kappa_array / gamma_array / delta_array
        print("Kappa/Gamma = {}".format(kappa_array/gamma_array))
        print ("Gamma/Delta = {}".format(gamma_array/delta_array))
        print ("Delta: {}".format(delta_array))
        outlier_index = index < treshold1
        sparse_region_index = np.logical_and(index >= treshold1, index < treshold2)
        dense_overlap_index = index >= treshold2


        if outlier_index>0:
            print ("Outlier region, index: {}".format(index))
            reward = (1/index[0] - 1/treshold1)*treshold1/(treshold1-1)*0.2+0.8    #scales values from 0.8 to 1
        elif sparse_region_index > 0:
            if (distances[:, 0]) < treshold3:
                print ("Sparse region with close neighbour: {}".format(index))
                reward = 0.0
            else:
                print ("Sparse region: {}".format(index))
                reward = (1/index[0] -1/treshold2)*(treshold1*treshold2)/(treshold2-treshold1)*0.3 + 0.5 #scales values from 0.5 to 0.8
        else:
            if (distances[:,0]) < treshold3:
                print ("Close neighbour: {}".format(index))
                reward = 0.0
            else:
                print ("Dense region: {}".format(index))   #scales values from 0 to 0.5
                reward = (1/index[0])*treshold2/2

        # reward = max(outlier + 0.8*sparse_region, outlier, sparse_region * 0.8) / (outlier + sparse_region + dense_overlap)


    #tresholds have to be set carefully

    return reward
