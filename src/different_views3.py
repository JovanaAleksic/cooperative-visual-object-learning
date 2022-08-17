from __future__ import division
import numpy as np
from overlap_measures1 import overlap_measures1


def different_views3(training_data, test_data, n_neighbors, neigh, treshold1, treshold2, treshold3):

    if len(training_data)<2:
        reward = 1.0
    elif len(training_data) <= n_neighbors and len(training_data)>=2:
        n=len(training_data)
        kappa1, gamma1, delta1, distances, indices = overlap_measures1(neigh, n-1, training_data, test_data)
        # kappa_array = np.asarray(kappa1)
        gamma_array = np.asarray(gamma1)
        delta_array = np.asarray(delta1)

        if gamma_array <= treshold1:
            # print ("Dense region")
            reward = gamma_array / treshold1 * 0.6
        elif gamma_array > treshold1 and (delta_array / gamma_array) < treshold2:
            if (distances[:, 0]) < treshold3:
                # print ("Sparse region with close neighbour")
                reward = (distances[:, 0] / treshold3) * 0.2 + 0.5
            else:
                # print ("Sparse region")
                reward = (gamma_array / treshold1) * 0.6 + 0.1
        else:
            if (distances[:, 0]) < treshold3:
                # print ("Close neighbour")
                reward = (distances[:, 0] / treshold3) * 0.2 + 0.5
            else:
                # print ("Outlier region")
                reward = (gamma_array / treshold1) * 0.8

    else:
        kappa1, gamma1, delta1, distances, indices = overlap_measures1(neigh, n_neighbors, training_data, test_data)
        # kappa_array = np.asarray(kappa1)
        gamma_array = np.asarray(gamma1)
        delta_array = np.asarray(delta1)

        if gamma_array <= treshold1:
            # print ("Dense region")
            reward = gamma_array/treshold1 * 0.6
        elif gamma_array > treshold1 and (delta_array/gamma_array) < treshold2:
            if (distances[:,0]) < treshold3:
                # print ("Sparse region with close neighbour")
                reward = (distances[:,0]/treshold3)*0.2 + 0.5
            else:
                # print ("Sparse region")
                reward = (gamma_array/treshold1) * 0.6 + 0.1
        else:
            if (distances[:, 0]) < treshold3:
                # print ("Close neighbour")
                reward = (distances[:,0]/treshold3)*0.2 + 0.5
            else:
                # print ("Outlier region")
                reward = (gamma_array/treshold1)*0.8
    return reward
