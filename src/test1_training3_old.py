import matplotlib.pyplot as plt
import numpy as np
import random
from get_views1 import get_views1
from sklearn.neighbors import KNeighborsClassifier
from confidence_calc import confidence_calc
from get_training_indexes import get_training_indexes
import pickle
from different_views2 import different_views2

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
objId_list=all
objId_test_list = all
# neuron_id = [200, 722]                                                          #set neuron_id=all for all neurons
neuron_id = all
n_neighbors = 5
f1 = 4                                                                             #first time confidence is computed
k = 5
limit_test_views = 50                      #10frames/sec x 5 sec for testing = 50
treshold = 0.8
treshold1 = 0.025                                                                 #maximum 10, 0 for not finiding new classes, 0 to leave out new objects
treshold2 = 0.5
# treshold1 = 5                                                                #maximum 10, 0 for not finiding new classes, 0 to leave out new objects
# treshold2 = 30
treshold1t = 0.35
treshold2t = 0.2
treshold3t = 0.15
repeat = 100
training_treshold = 0.7
step = 6

# z = 1
# pickle.dump(z, open("Strategies/TestTraining/number.pickle", "wb"))

z = pickle.load(open("Strategies/TestTraining/number.pickle", "rb"))

z += 1

pickle.dump(z, open("Strategies/TestTraining/number.pickle", "wb"))

name = "Strategies/TestTraining/TestTraining{}.pickle"
pic = "Strategies/TestTraining/TestTraining{}.png"
with open(name.format(z), 'w') as f:
    pickle.dump([objId_list, objId_test_list, neuron_id, n_neighbors, f1, k, limit_test_views, repeat, treshold, treshold1, treshold2, treshold1t, treshold2t, treshold3t, training_treshold, step], f)

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

new_classes = []
for h in range(len(objId_test_list)):
    if objId_test_list[h] not in objId_list:
        new_classes.append(objId_test_list[h])

for t in range(repeat):
    print ("Run: {}".format(t))
    classification_rate = []
    training_labels = []
    training_index = random.sample(range(total_views), 1)[0]
    total_training_data, view_test_list_main = get_training_indexes(training_index, k)

    for e in number_of_training_samples_list:
        predicted_classes = []
        test_labels_total = []
        # print ("Training batch: {}".format(e))

        # #creates list of indexes of views for training
        # a = random.sample(range(len(total_training_data) - e), 1)[0]
        # view_training_list = total_training_data[a:(a + e)]
        #
        # #gets training data and training labels
        # training_data, training_labels = get_views1(objId_list, view_training_list, data_location, neuron_id)

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
                different_reward = different_views2(temporary_training, temporary_test, n_neighbors, neigh1, treshold1t, treshold2t, treshold3t)

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
            confidence1, label1 = confidence_calc(test_data, training_data, training_labels, neigh, n_neighbors, treshold1, treshold2)

            if side_selector[0] == 0:
                view_test_list = view_test_list_main[g:(g+k)]
            else:
                view_test_list = view_test_list_main[(g-k):g][::-1]

            test_data, test_labels = get_views1([objId], view_test_list, data_location, neuron_id)
            confidence2, label2 = confidence_calc(test_data, training_data, training_labels, neigh, n_neighbors, treshold1, treshold2)

            if confidence2 > treshold:
                predicted_classes.append(label2)

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
                    confidence2, label2 = confidence_calc(test_data, training_data, training_labels, neigh, n_neighbors,
                                                                  treshold1, treshold2)
                predicted_classes.append(label2)


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
        classification_rate.append((np.sum(match_classes_array)) / float(len(match_classes_array) - b))
        # classification_rate.append((np.sum(match_classes_array) + b)/float(len(match_classes_array)))
        # print classification_rate
        # classification_rate.append(np.sum(match_classes_array) / float(len(match_classes_array)))
        # print ("Classification rate: {}".format(classification_rate))

    classification_rate_average.append(classification_rate)
    # print ("Classification rate average: {}".format(classification_rate_average))

classification_rate_average = np.sum(classification_rate_average, 0)/float(repeat)
# print ("Classification rate average: {}".format(classification_rate_average))


name_class = "Strategies/TestTraining/TestTraining_Rate{}.pickle"
with open(name_class.format(z), 'w') as f:
    pickle.dump([classification_rate_average], f)

#plot
plt.figure()
plt.plot(number_of_training_samples_list, classification_rate_average, '-bo')
plt.xlabel("Number of training samples per class")
plt.ylabel("Correct classification rate")
plt.title("Test+Training: Average over {} runs, {} objects, {} neurons, {} frames, {} neighbors".format(repeat, len(objId_list), len(neuron_id), k, n_neighbors))
plt.ylim([0, 1])
plt.savefig(pic.format(z))
# plt.show()