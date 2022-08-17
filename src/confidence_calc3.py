import numpy as np
from overlap_measures import overlap_measures
from overlap_detection_3 import overlap_detection_3
from actions import confidence_computation

def confidence_calc3(test_data,training_data, training_labels, neigh, n_neighbors, treshold3, perse, local_h):

    if len(training_data) == 0:
        confidence = 1.0
        label = 0
    else:
        s = []
        # print test_data.shape
        if n_neighbors > len(training_data):
            n_neighbors = len(training_data)

        kappa1, gamma1, delta1, indices = overlap_measures(neigh, n_neighbors, training_data, test_data)
        outlier, sparse_region, dense_overlap, index_total = \
            overlap_detection_3(treshold3, kappa1, gamma1, delta1, perse, local_h)
        confidence_approx, new, measure = confidence_computation(outlier, sparse_region, dense_overlap)

        if new == 1:
            confidence = confidence_approx
            label = 0

        else:
            for i in range(len(index_total)):
                if index_total[i] == 1:
                    for j in range(n_neighbors):
                        ix = indices[i][j]
                        s.append(training_labels[ix])  #
            cles_overlap = list(set(s))
            nb = len(cles_overlap)
            pro = np.zeros((nb))

            if nb == 1:
                confidence = confidence_approx
                label = s[0]

            else:
                for i in range(int(len(s) / n_neighbors)):
                    c = len(list(set(s[(i * n_neighbors):((i + 1) * n_neighbors)])))

                    if c == 1:
                        v = cles_overlap.index(s[i * n_neighbors])
                        pro[v] += 1

                    elif c == pro.shape[0]:
                        pro += 1

                    else:  # This is the case where you have for certain sample 2 neighboring classes but overall there is more overlap classe
                        p = list(set(s[(i * n_neighbors):((i + 1) * n_neighbors)]))
                        for j in p:
                            v = cles_overlap.index(j)
                            pro[v] += 1

                if outlier > 0:
                    confidence = ((max(pro.max(), outlier)) / (outlier + sparse_region + dense_overlap) - 1 / (
                    nb + 1)) * ((nb + 1) / nb)

                else:
                    confidence = ((max(pro.max(), outlier)) / (outlier + sparse_region + dense_overlap) - 1 /
                        nb) * (nb / (nb-1))

                if outlier > pro.max():
                    label = 0

                else:
                    h = np.where(pro == pro.max())

                    if len(h[0]) == 1:
                        label = cles_overlap[pro.argmax()]

                    else:
                        label = max(s, key=s.count)

                if confidence > 1.0:
                    confidence = 1.0

    return confidence, label