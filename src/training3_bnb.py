#DESCRIPTION
#This is combination of training_strategy3 and testing in nostrategy3
########################################################################################################################
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import random
from get_views1 import get_views1
from sklearn.neighbors import KNeighborsClassifier
from get_training_indexes import get_training_indexes
import pickle
from different_views import different_views
from different_views2 import different_views2
from get_views2 import get_views2
########################################################################################################################
#PARAMETERS

total_views = 1200
classification_rate_average = []
# number_of_training_samples_list = [10, 50, 100, 200, 500]
number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]
training_strategy = "random"
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"
# objId_list = random.sample(range(126), 10)
# objId_list = [6, 7, 70, 55, 81, 2, 40, 20, 85, 90]
# objId_test_list = [6, 7, 70, 55, 81, 2, 40, 21, 86, 91]
#set objId_list=all for all objects
# objId_list=[6,90]
objId_list=all
objId_test_list = all
# neuron_id = [200, 722]                                                        #set neuron_id=all for all neurons
neuron_id = all
n_neighbors = 5
k = 5
repeat = 100
training_treshold = 0.7                                                        #MUST BE under 0.8, doesn't make sense otherwise
# treshold1 = 6
# treshold2 = 30
# treshold3=5
step = 6

# treshold1 = 0.35
# treshold2 = 0.2
# treshold3 = 0.15

treshold1 = 0.35
treshold2 = 0.2
treshold3 = 0.15

# z = 1
# pickle.dump(z, open("Strategies/Training3/number.pickle", "wb"))

z = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training3/number.pickle", "rb"))

z += 1

pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Strategies/Training3/number.pickle", "wb"))

name = "/hri/storage/user/jradojev/Version1/Strategies/Training3+bnb/Training{}.pickle"
pic = "/hri/storage/user/jradojev/Version1/Strategies/Training3+bnb/Training{}.png"
with open(name.format(z), 'w') as f:
    pickle.dump([objId_list, objId_test_list, neuron_id, n_neighbors, k, repeat, training_treshold, treshold1, treshold2, treshold3, step], f)

########################################################################################################################
#MAIN

if objId_list == all:
    objId_list = range(1, 127)


if objId_test_list == all:
    objId_test_list = range(1, 127)

if neuron_id == all:
    neuron_id = range(1000)

k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))

for t in range(repeat):
    total_training_data, view_test_list_main = [], []
    print ("Run: {}".format(t))
    classification_rate = []
    training_labels = []
    training_index = random.sample(range(total_views), 1)[0]
    total_training_data, view_test_list_main = get_training_indexes(training_index, k)

    # print len(total_training_data)
    # print len(view_test_list_main)

    for e in number_of_training_samples_list:
        # print("Training number: {}".format(e))
        for objId in objId_list:
            # print("Object: {}".format(objId))
            view_training_list = []
            temporary_training = []
            temporary_labels = []
            main = []
            neigh1 = 0
            j = random.sample(range(1000), 1)[0]
            for i in range(1000):   #has to be checked! not good! sometimes doesn't add enough data
                # print i
                temporary_test_index = [int(total_training_data[(j+i*step) % 1000])]
                # print ("Temp:{}".format(temporary_test_index))
                temporary_test, test_labels = get_views1([objId], temporary_test_index, data_location, neuron_id)
                different_reward = different_views2(temporary_training, temporary_test, n_neighbors, neigh1, treshold1, treshold2, treshold3)

                if different_reward > training_treshold:
                    # view_training_list.append(temporary_test_index[0])
                    xs = []
                    filename = data_location.format(
                        objId, temporary_test_index[0])
                    array = np.load(filename)['data']
                    for l in range(len(neuron_id)):
                        xs.append(array[0][neuron_id[l]])
                    temporary_labels.extend([objId])
                    main.append(xs)
                    temporary_training = np.asarray(main)

                # temporary_training, temporary_labels = get_views1([objId], view_training_list, data_location, neuron_id)

                if len(temporary_labels) == e:
                    break

                neigh1 = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(temporary_training, temporary_labels)

            if objId_list.index(objId) != 0:
                training_data = np.append(training_data, temporary_training, 0)
                training_labels.extend(temporary_labels)
            else:
                training_data = temporary_training
                training_labels = temporary_labels

            # print("Added {} training samples".format(len(temporary_labels)))  #

        neigh = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(training_data, training_labels)

        # g = random.sample(range(k, len(view_test_list_main)-k), 1)[0]
#         # print g
#         side_selector = random.sample(range(2), 1)
#         # print side_selector
#
#         if side_selector[0] == 0:
#             view_test_list = view_test_list_main[g:(g+k)]
#         else:
#             view_test_list = view_test_list_main[(g-k+1):(g+1)]
#
#         # print view_test_list
#         #gets test data and real test labels
#         test_data, test_labels = get_views1(objId_test_list, view_test_list, data_location, neuron_id)
#         predicted_classes = neigh.predict(test_data)
#
#         predicted_classes = predicted_classes.tolist()
#         predicted_classes_total = []
#         test_labels_total = []
#
#         for i in range(len(objId_list)):
#             predicted_classes_total.append(max(predicted_classes[i*k:(i+1)*k], key=predicted_classes[i*k:(i+1)*k].count))
#             test_labels_total.append(max(test_labels[i*k:(i+1)*k], key=test_labels[i*k:(i+1)*k].count))
#
#         # print predicted_classes_total
#         # print test_labels_total
#
#         #boolean array which says for every test instance is prediction equal to real label
#         match_classes_array = np.asarray(predicted_classes_total) == np.asarray(test_labels_total)
#         classification_rate.append(np.sum(match_classes_array)/float(len(match_classes_array)))
#
#     classification_rate_average.append(classification_rate)
#     # print ("Classification rate average: {}".format(classification_rate_average))
#
# classification_rate_average = np.sum(classification_rate_average, 0) / float(repeat)
# # print ("Classification rate average: {}".format(classification_rate_average))

        f = range(50, 150)
        g = random.sample(f, 1)[0]
        # print view_test_list
        #gets test data and real test labels
        test_data, test_labels = get_views2(objId_test_list, view_test_list_main, k_array, data_location, neuron_id, g, t, number_of_training_samples_list, e)
        # print ("Shape test data: {}".format(test_data.shape))
        predicted_classes = neigh.predict(test_data)

        # proba_total=0
        # for objId in objId_test_list:
        #     proba = int(k_array[0][t][number_of_training_samples_list.index(e)][objId - 1])
        #     proba_total+=proba

        # print ("Total number of views: {}".format(proba_total))
        predicted_classes = predicted_classes.tolist()
        # print ("Predicted classes length: {}".format(len(predicted_classes)))
        predicted_classes_total = []
        test_labels_total = []

        for objId in objId_test_list:
            k = int(k_array[0][t][number_of_training_samples_list.index(e)][objId_test_list.index(objId)-1])
            # print objId
            predicted_classes_total.append(max(predicted_classes[0:k], key=predicted_classes[0:k].count))
            test_labels_total.append(max(test_labels[0:k], key=test_labels[0:k].count))
            predicted_classes = predicted_classes[k::]
            test_labels=test_labels[k::]
            # print len(predicted_classes)

        # print predicted_classes_total
        # print test_labels_total

        #boolean array which says for every test instance is prediction equal to real label
        match_classes_array = np.asarray(predicted_classes_total) == np.asarray(test_labels_total)
        classification_rate.append(np.sum(match_classes_array)/float(len(match_classes_array)))

    classification_rate_average.append(classification_rate)
    # print ("Classification rate average: {}".format(classification_rate_average))

classification_rate_average = np.sum(classification_rate_average, 0) / float(repeat)
# print ("Classification rate average: {}".format(classification_rate_average))


name_class = "/hri/storage/user/jradojev/Version1/Strategies/Training3+bnb/Training_Rate{}.pickle"
with open(name_class.format(z), 'w') as f:
    pickle.dump([classification_rate_average], f)

#plot
plt.figure()
plt.plot(number_of_training_samples_list, classification_rate_average, '-bo')
plt.xlabel("Number of training samples per class")
plt.ylabel("Correct classification rate")
plt.title("Training3+bnb: Average over {} runs, {} objects, {} frames, {} neurons, {} neighbors".format(repeat, len(objId_list), k, len(neuron_id), n_neighbors, training_strategy))
plt.ylim([0, 1])
plt.savefig(pic.format(z))
# plt.show()



