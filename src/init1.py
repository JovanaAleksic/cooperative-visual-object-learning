import matplotlib.pyplot as plt
import numpy as np
import random
from get_views1 import get_views1
from sklearn.neighbors import KNeighborsClassifier
from get_training_indexes import get_training_indexes
import pickle
from overlap_measures_temp import overlap_measures_temp
########################################################################################################################
#PARAMETERS

total_views = 1200
number_of_training_samples_list = [6]
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"
objId_list = random.sample(range(1,126), 1)
# objId_list = [6, 7, 70, 55, 81, 2, 40, 20, 85, 90]
# objId_test_list = [6, 7, 70, 55, 81, 2, 40, 21, 86, 91] #set objId_list=all for all objects
# objId_list=[6,90]
# objId_list = [50]
objId_test_list = objId_list
# neuron_id = [200, 722]
# neuron_id = [200, 722, 300, 999]
# neuron_id = [200, 722, 300, 999, 500, 100]        #set neuron_id=all for all neurons
neuron_id = all
n_neighbors = 2
k = 5
repeat = 1

########################################################################################################################
#FUNCTION SUBSETS
def findMinSubset(gamma1, i, sumCalculated, sumTotal):
    if i == 0:
        return abs((sumTotal - sumCalculated) - sumCalculated)
    else:
        return max(findMinSubset(gamma1, i - 1, sumCalculated + gamma1[i - 1], sumTotal),
                   findMinSubset(gamma1, i - 1, sumCalculated, sumTotal))


########################################################################################################################
#MAIN

if objId_list == all:
    objId_list = range(1, 127)

if objId_test_list == all:
    objId_test_list = range(1, 127)

if neuron_id == all:
    neuron_id = range(1000)

for t in range(repeat):
    # k = random.sample(range(5, 50, 5), 1)[0]
    total_training_data, view_test_list_main = [], []
    print ("Run: {}".format(t))
    classification_rate = []
    training_index = random.sample(range(total_views), 1)[0]
    total_training_data, view_test_list_main = get_training_indexes(training_index, k)

    # print len(total_training_data)
    # print len(view_test_list_main)

    for e in number_of_training_samples_list:


        # a = random.sample(range(len(total_training_data)-e), 1)[0]
        # view_training_list = total_training_data[0::(len(total_training_data)/e)]
        view_training_list = total_training_data

        #gets training data and training labels
        training_data, training_labels = get_views1(objId_list, view_training_list, data_location, neuron_id)
        neigh = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(training_data, training_labels)

        view_test_list = view_training_list

        # print len(view_test_list)
        #gets test data and real test labels
        test_data, test_labels = get_views1(objId_test_list, view_test_list, data_location, neuron_id)
        kappa1, gamma1, delta1, indices = overlap_measures_temp(neigh, n_neighbors, training_data, test_data)
########################################################################################################################
#ONLY FOR ALL VIEWS STRATEGY
        # sumTotal = 0
        # for i in range(len(gamma1)):
        #     sumTotal += gamma1[i]
        # sumTotal_average=sumTotal/float(len(gamma1))
        #
        # max_sum = findMinSubset(gamma1, len(gamma1), 0, sumTotal)
        # print ("Max sum: {}".format(max_sum))
        #

########################################################################################################################

        gamma_average = sum(gamma1)/len(gamma1)
        print ("Gamma average: {}".format(gamma_average))
        alfa = 0.95
        treshold1=gamma_average*alfa
        size = sum(gamma1 < treshold1) / float(len(gamma1)) * 100
        while size > 30:
            alfa = alfa - 0.05
            treshold1 = gamma_average * alfa
            size=sum(gamma1 < treshold1)/float(len(gamma1))*100

        print ("Alfa: {}".format(alfa))
        print ("Size: {}%".format(size))
        delta_average = sum(delta1) / len(delta1)
        print ("Delta average: {}".format(delta_average))
        # print ("Gamma: {}".format(gamma1))
        # print ("Gamma average: {}".format(sum(gamma1)/float(len(view_training_list))))
        # print ("Delta: {}".format(delta1))
        # print ("Delta average: {}".format(sum(delta1)/float(len(view_training_list))))



