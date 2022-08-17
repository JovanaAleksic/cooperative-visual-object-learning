from __future__ import division
from different_views3 import different_views3
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random
from get_views1 import get_views1


def phase_training(neigh1, training_data, training_labels, n_neighbors, treshold1, treshold2, treshold3, training_treshold, data_location, neuron_id, pick_object, gog, gamma_average, a_p, c_p):
    main = []
    mintr = 150
    maxtr = 200
    temporary_labels = []
    temporary_training = []
    training_reward = []
    joj = random.sample(range(1000-mintr), 1)[0]
    joj1 = random.sample(range(1000), 1)[0]
    step = 1

    # bottom = min(joj1, joj)
    bottom = joj
    # upp = max(joj, joj1)
    # diff = upp-bottom
    diff = mintr

    if diff < mintr:
        diff = mintr

    if diff > maxtr:
        diff=maxtr

    for i in range(0, diff, step):
        # print ("bottom + i: {}".format(bottom+i))
        temporary_test_index = [bottom+i*step]
        # print ("Temp_test_index: {}".format(temporary_test_index))
        temporary_test, test_labels = get_views1([pick_object], temporary_test_index, data_location, neuron_id)

        different_reward = different_views3(temporary_training, temporary_test, n_neighbors, neigh1, treshold1, treshold2, treshold3)

        training_reward.append(different_reward)

        if different_reward > training_treshold:
            # view_training_list.append(temporary_test_index[0])
            xs = []
            filename = data_location.format(pick_object, temporary_test_index[0])
            array = np.load(filename)['data']
            for li in range(len(neuron_id)):
                xs.append(array[0][neuron_id[li]])
            temporary_labels.append(pick_object)
            leny = len(temporary_labels)
            main.append(xs)
            temporary_training = np.asarray(main)
            if 100 >= leny > n_neighbors:
                treshold1 = (gog - gog * (leny - n_neighbors) / float(100 - n_neighbors) * (1 - a_p / gog)) * gamma_average
                treshold3 = c_p * treshold1

        if 2 <= len(temporary_labels) <= n_neighbors:
            neigh1 = KNeighborsClassifier(len(temporary_labels) - 1, algorithm='brute').fit(temporary_training, temporary_labels)
        else:
            neigh1 = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(temporary_training, temporary_labels)

    if len(training_data) != 0:
        training_data = np.append(training_data, temporary_training, 0)
        training_labels.extend(temporary_labels)
    else:
        training_data = temporary_training
        training_labels = temporary_labels

    print ("Added: {}".format(len(temporary_labels)))
    if n_neighbors < len(training_data):
        neigh1 = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(training_data, training_labels)
    else:
        neigh1 = KNeighborsClassifier(len(training_data), algorithm='brute').fit(training_data, training_labels)
    return training_data, neigh1, training_reward, training_labels, len(temporary_labels)