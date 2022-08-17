import matplotlib.pyplot as plt
import numpy as np
import random
from get_views1 import get_views1
from sklearn.neighbors import KNeighborsClassifier
from get_training_indexes import get_training_indexes
import pickle
from different_views import different_views
from different_views2 import different_views2
########################################################################################################################
#PARAMETERS

data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"

neuron_id = [200, 722]                                                        #set neuron_id=all for all neurons
n_neighbors = 5
k = 5
treshold1 = 0.35
treshold2 = 0.2
# repeat = 500
treshold3 = 0.15   #this has to be set



########################################################################################################################
#MAIN

view_training_list = range(0, 100)

objId = 100
training_data, training_labels = get_views1([objId], view_training_list, data_location, neuron_id)
neigh = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(training_data, training_labels)

view_test_list = random.sample(range(1000, 1200), 1)
print ("View number: {}".format(view_test_list))
test_data, test_labels = get_views1([objId], view_test_list, data_location, neuron_id)

# different_reward = different_views(training_data, test_data, n_neighbors, neigh, treshold1, treshold2, treshold3)
different_reward = different_views2(training_data, test_data, n_neighbors, neigh, treshold1, treshold2, treshold3)
print ("Reward: {}".format(different_reward))

#plot
plt.figure()
plt.plot(training_data[:, 0], training_data[:, 1], 'b+')
plt.plot(test_data[:, 0], test_data[:, 1], 'ro')
# plt.ylim([5, 20])
# plt.xlim([-7, 5])
plt.show()


