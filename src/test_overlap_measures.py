from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from get_views import get_views
from overlap_measures import overlap_measures
from overlap_detection import overlap_detection
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from get_views1 import get_views1
from overlap_detection_2 import overlap_detection_2
import pickle
from overlap_detection_3 import overlap_detection_3

data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"
objId1 = 6
objId2 = 10
view_number = 800
treshold1 = 1.5  #maximum 10
treshold2 = 30
n_neighbors = 5
neuron_id = [200, 900]
# neuron_id =  all
perse = 0.7
local_h = 1

view_training_list = range(1200)
view_test_list = range(1200)

treshold1 = pickle.load(open("Strategies/Training6/Tresholds/treshold1_133.pickle", "rb"))[0]                                                              #maximum 10, 0 for not finiding new classes, 0 to leave out new objects
treshold2 = pickle.load(open("Strategies/Training6/Tresholds/treshold2_133.pickle", "rb"))[0]
treshold3 = pickle.load(open("Strategies/Training6/Tresholds/treshold3_133.pickle", "rb"))[0]

print treshold3


training_data, training_labels = get_views1([objId1], view_training_list, data_location, neuron_id)
neigh = NearestNeighbors(n_neighbors).fit(training_data)

test_data, test_labels = get_views1([objId2], view_test_list, data_location, neuron_id)
kappa1, gamma1, delta1, indices = overlap_measures(neigh, n_neighbors, training_data, test_data)
# outlier, sparse_region, dense_overlap, index_overlap_total =\
#     overlap_detection_2(treshold1, treshold2, kappa1, gamma1, delta1)

outlier, sparse_region, dense_overlap, index_overlap_total =\
    overlap_detection_3(treshold3, kappa1, gamma1, delta1, perse, local_h)



percent = (sparse_region+dense_overlap)/(outlier + dense_overlap + sparse_region)
percent *=100

print("Overlap extent: {} %".format(round(percent , 2)))
print("Dense: {} %".format(round((dense_overlap/(outlier + dense_overlap + sparse_region)), 4)*100))
print("Sparse: {} %".format(round((sparse_region/(outlier + dense_overlap + sparse_region)), 4)*100))
print("Outlier: {} %".format(round(outlier/(outlier + dense_overlap + sparse_region), 4)*100))

fig = plt.figure(1)
plt.plot(test_data[:,0], test_data[:,1], 'b+')
plt.plot(training_data[:,0], training_data[:,1], 'r+')
plt.show()