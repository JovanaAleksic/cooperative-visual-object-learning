#DESCRIPTION
#this test strategy is derived from training6/training5 treshold which discards third of the samples
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
from get_views1 import get_views1
from sklearn.neighbors import KNeighborsClassifier
from confidence_calc2 import confidence_calc2
from get_training_indexes import get_training_indexes
import pickle

########################################################################################################################
#PARAMETERS

total_views = 1200
# number_of_training_samples_list = [10, 50, 100, 200, 500]  #100 (10 frames/sec x 10sec for training) makes sense to be minimal
# number_of_training_samples_list = [100]
number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"
# objId_list = random.sample(range(126), 10)
# objId_list = [6, 7, 70, 55, 81, 2, 40, 20, 85, 90]                              #set objId_list=all for all objects
# objId_test_list = [6, 7, 70, 55, 81, 2, 40, 21, 86, 91]
# objId_list=[6,90]
# objId_list = random.sample(range(1, 126), 100)
# objId_test_list = random.sample(range(1, 126), 100)
objId_list = all
objId_test_list = all
neuron_id = random.sample(range(0, 1000), 2)                                                        #set neuron_id=all for all neurons
# neuron_id = all
n_neighbors = 5
f1 = 4                                                                             #first time confidence is computed
k = 5
limit_test_views = 50                      #10frames/sec x 5 sec for testing = 50
treshold = 0.8
# treshold1 = 0.025                                                                 #maximum 10, 0 for not finiding new classes, 0 to leave out new objects
# treshold2 = 0.5 - 0.7
treshold1 = pickle.load(open("Strategies/Training6/Tresholds/treshold1.pickle", "rb"))                                                               #maximum 10, 0 for not finiding new classes, 0 to leave out new objects
treshold2 = pickle.load(open("Strategies/Training6/Tresholds/treshold2.pickle", "rb"))
repeat = 100

local_h = 3
local_g = 1


# z = 1
# pickle.dump(z, open("Strategies/Test/number.pickle", "wb"))

z = pickle.load(open("Strategies/Test/number.pickle", "rb"))

z += 1

pickle.dump(z, open("Strategies/Test/number.pickle", "wb"))

name = "Strategies/Test2/Test{}.pickle"
pic = "Strategies/Test2/Test{}.png"
with open(name.format(z), 'w') as f:
    pickle.dump([objId_list, objId_test_list, neuron_id, n_neighbors, f1, k, limit_test_views, repeat, treshold, treshold1, treshold2], f)

########################################################################################################################


def length(s):                                                                  #number of different objects in a list
    return len(list(set(s)))
classification_rate_average = []


########################################################################################################################
#MAIN

if objId_list == all:
    objId_list = range(1, 127)

if objId_test_list == all:
    objId_test_list = range(1, 127)

if neuron_id == all:
    neuron_id = range(1000)

k_array=np.zeros((repeat, len(number_of_training_samples_list), len(objId_test_list)))

new_classes = []
for h in range(len(objId_test_list)):
    if objId_test_list[h] not in objId_list:
        new_classes.append(objId_test_list[h])

for t in range(repeat):
    # objId_list = random.sample(range(1, 126), 100)
    # objId_test_list = random.sample(range(1, 126), 100)
    neuron_id = random.sample(range(0, 1000), 2)
    print ("Run: {}".format(t))
    classification_rate = []
    training_labels = []
    training_index = random.sample(range(total_views), 1)[0]
    total_training_data, view_test_list_main = get_training_indexes(training_index, k)

    for e in number_of_training_samples_list:
        predicted_classes = []
        test_labels_total = []
        # print ("Training batch: {}".format(e))

        #creates list of indexes of views for training
        a = random.sample(range(len(total_training_data) - e), 1)[0]
        view_training_list = total_training_data[a:(a + e)]

        #gets training data and training labels
        training_data, training_labels = get_views1(objId_list, view_training_list, data_location, neuron_id)
        neigh = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(training_data, training_labels)

        g = random.sample(range(k, len(view_test_list_main)-k), 1)[0]
        # print g
        side_selector = random.sample(range(2), 1)

        for objId in objId_test_list:
            # print ("Object: {}".format(objId))
            test_labels_total.append(objId)


            if side_selector[0] == 0:
                view_test_list = view_test_list_main[g: (g + f1)]
            else:
                view_test_list = view_test_list_main[(g - f1):g][::-1]

            test_data, test_labels = get_views1([objId], view_test_list, data_location, neuron_id)
            confidence1, label1 = confidence_calc2(test_data, training_data, training_labels, neigh, n_neighbors, treshold1, treshold2, local_h, local_g)

            if side_selector[0] == 0:
                view_test_list = view_test_list_main[g:(g+k)]
            else:
                view_test_list = view_test_list_main[(g-k):g][::-1]

            test_data, test_labels = get_views1([objId], view_test_list, data_location, neuron_id)
            confidence2, label2 = confidence_calc2(test_data, training_data, training_labels, neigh, n_neighbors, treshold1, treshold2,local_h, local_g)

            if confidence2 > treshold:
                predicted_classes.append(label2)
                k_array[t][number_of_training_samples_list.index(e)][objId_test_list.index(objId)]=k

            else:
                if confidence2 < confidence1:
                    side_selector[0] = not(side_selector[0])
                # print ("View test list elif: {}".format(len(view_test_list)))

                while confidence2 < treshold and (len(view_test_list)<limit_test_views):
                    # print ("In while loop: {}".format(len(view_test_list)))
                    l = random.sample(range(k, len(view_test_list_main)-k), 1)[0]
                    if side_selector[0] == 0:
                        view_test_list.extend(view_test_list_main[l:l+k])
                    else:
                        view_test_list.extend(view_test_list_main[l-k:l][::-1])

                    test_data, test_labels = get_views1([objId], view_test_list, data_location, neuron_id)
                    confidence2, label2 = confidence_calc2(test_data, training_data, training_labels, neigh, n_neighbors,
                                                                  treshold1, treshold2, local_h, local_g)
                predicted_classes.append(label2)
                k_array[t][number_of_training_samples_list.index(e)][objId_test_list.index(objId)] = len(view_test_list)


        b=0
        #boolean array which says for every test instance is prediction equal to real label
        match_classes_array = np.asarray(predicted_classes) == np.asarray(test_labels_total)
        # print predicted_classes
        # print test_labels_total
        # print match_classes_array

        new_predicted_classes = np.asarray(predicted_classes) == 0
        # print new_predicted_classes
        # print range(len(test_labels_total))
        for g in range(len(test_labels_total)):
            if new_predicted_classes[g] == 1:
                if test_labels_total[g] in new_classes:
                    b += 1


        # print b
        # print new_classes
        classification_rate.append((np.sum(match_classes_array))/float(len(match_classes_array) - b))
        # classification_rate.append((np.sum(match_classes_array) + b) / float(len(match_classes_array)))  #my first method
        # print classification_rate
        # classification_rate.append(np.sum(match_classes_array) / float(len(match_classes_array)))
        # print ("Classification rate: {}".format(classification_rate))

    classification_rate_average.append(classification_rate)
    # print ("Classification rate average: {}".format(classification_rate_average))

classification_rate_average = np.sum(classification_rate_average,0)/float(repeat)
# print ("Classification rate average: {}".format(classification_rate_average))


name_class = "Strategies/Test2/Test_Rate{}.pickle"
with open(name_class.format(z), 'w') as f:
    pickle.dump([classification_rate_average], f)

name_k = "Strategies/Test2/k{}.pickle"
with open(name_k.format(z), 'w') as f:
    pickle.dump([k_array], f)

#plot
plt.figure()
plt.plot(number_of_training_samples_list, classification_rate_average, '-bo')
plt.xlabel("Number of training samples per class")
plt.ylabel("Correct classification rate")
plt.title("Test: {} r, {} trobj, {} teobj, {} neuro, {} local_h, {} g,{} neighbors".format(repeat, len(objId_list), len(objId_test_list), len(neuron_id), local_h, local_g,  n_neighbors))
plt.ylim([0, 1])
plt.savefig(pic.format(z))
# plt.show()