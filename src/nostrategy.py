import matplotlib.pyplot as plt
import numpy as np
import random
from get_views1 import get_views1
from sklearn.neighbors import KNeighborsClassifier

########################################################################################################################
#PARAMETERS

classification_rate = []
number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"
# objId_list = random.sample(range(126), 10)
objId_list = [6, 7, 70, 55, 81, 2, 40, 20, 85, 90]                              #set objId_list=all for all objects
# objId_list=[6,90]
# objId_list=all
# neuron_id = [200, 722, 1, 500, 900, 100]                                                          #set neuron_id=all for all neurons
neuron_id = all
n_neighbors = 5
view_test_list = range(1000, 1200, 2)

########################################################################################################################
#MAIN

if objId_list == all:
    objId_list = range(1, 127)

if neuron_id == all:
    neuron_id = range(1000)

for e in number_of_training_samples_list:

    #creates list of indexes of views for training
    view_training_list = []
    for i in range(e):
        view_training_list.append(int(i*(1000/float(e))))

    #gets training data and training labels
    training_data, training_labels = get_views1(objId_list, view_training_list, data_location, neuron_id)
    neigh = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(training_data, training_labels)


    #gets test data and real test labels
    test_data, test_labels = get_views1(objId_list, view_test_list, data_location, neuron_id)
    predicted_classes = neigh.predict(test_data)

    #boolean array which says for every test instance is prediction equal to real label
    match_classes_array = np.asarray(predicted_classes) == np.asarray(test_labels)
    classification_rate.append(sum(match_classes_array)/float(len(match_classes_array)))


#plot
plt.figure()
plt.plot(number_of_training_samples_list, classification_rate, '-bo')
plt.xlabel("Number of training samples per class")
plt.ylabel("Correct classification rate")
plt.title("Baseline_1_sample: {} objects, {} neurons, {} test points per class, {} neighbors".format(len(objId_list), len(neuron_id), len(view_test_list), n_neighbors))
plt.ylim([0, 1])
plt.show()



