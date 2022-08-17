import numpy as np

def overlap_measures_temp(neigh, n_neighbors, feature_memory, test):
    distances, indices = neigh.kneighbors(test)
    distances = distances[:,1::]
    indices = indices[:,1::]
    # print distances.shape
    # print indices.shape
    kappa = distances[:, n_neighbors-2]
    gamma = np.sum(distances, 1)/(n_neighbors-1)

    delta_s = []
    for i in range(n_neighbors-1):
        delta_s.append(feature_memory[indices[:, i]])
    delta = np.linalg.norm(np.sum(test - delta_s, 0), ord=None, axis=1) / (n_neighbors-1)
    return kappa, gamma, delta, indices
