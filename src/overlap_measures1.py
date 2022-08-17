import numpy as np

def overlap_measures1(neigh, n_neighbors, feature_memory, test):
    distances, indices = neigh.kneighbors(test)
    kappa = distances[:, n_neighbors-1]
    gamma = np.sum(distances, 1)/n_neighbors

    delta_s = []
    for i in range(n_neighbors):
        delta_s.append(feature_memory[indices[:, i]])
    delta = np.linalg.norm(np.sum(test - delta_s, 0), ord=None, axis=1) / n_neighbors
    return kappa, gamma, delta, distances, indices
