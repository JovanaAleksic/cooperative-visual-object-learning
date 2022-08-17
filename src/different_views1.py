from __future__ import division
import numpy as np
from overlap_measures1 import overlap_measures1


def different_views1(training_data, test_data, n_neighbors, neigh, treshold1, average):

    a = len(training_data)
    if a < n_neighbors:
        reward = 1.0
        average1 = 0
    elif a == n_neighbors:
        kappa1, gamma1, delta1, distances, indices = overlap_measures1(neigh, n_neighbors, training_data, test_data)
        gamma_array = np.asarray(gamma1)
        reward = 1.0
        average1 = gamma_array
    else:
        kappa1, gamma1, delta1, distances, indices = overlap_measures1(neigh, n_neighbors, training_data, test_data)

        kappa_array = np.asarray(kappa1)
        gamma_array = np.asarray(gamma1)
        delta_array = np.asarray(delta1)


        index = kappa_array / gamma_array / delta_array
        # print("Kappa/Gamma = {}".format(kappa_array/gamma_array))
        # print ("Gamma/Delta = {}".format(gamma_array/delta_array))
        # print ("Delta: {}".format(delta_array))
        outlier_index = index < treshold1



        if outlier_index>0:
            # print ("Outlier region, index: {}".format(index))
            reward= (1/index[0] - 1/treshold1)*treshold1/(treshold1-1)*0.2+0.8    #scales values from 0.8 to 1
            average1 = (average * a + gamma_array) / (a + 1)
        else:
            if gamma_array > average:
                reward = 1.0
                average1 = (average * a + gamma_array) / (a + 1)
            else:
                reward = 0.0
                average1 = average


    return reward, average1
