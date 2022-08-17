import matplotlib.pyplot as plt
import numpy as np
import random
from get_views1 import get_views1
from sklearn.neighbors import KNeighborsClassifier
from confidence_calc import confidence_calc
from get_training_indexes import get_training_indexes
import pickle
########################################################################################################################

def length(s):                                                                  #number of different objects in a list
    return len(list(set(s)))

########################################################################################################################
#PARAMETERS

total_views = 1200
# number_of_training_samples_list = [10]
number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"
n_neighbors = 5
f1 = 4                                                                             #first time confidence is computed
k = 5
limit_test_views = 100                      #10frames/sec x 5 sec for testing = 50
treshold = 0.8
# treshold1_test = 0.025                                                                 #maximum 10, 0 for not finiding new classes, 0 to leave out new objects
# treshold2_test = 0.5 - 0.7
# treshold1_test = 0.015                                                                 #maximum 10, 0 for not finiding new classes, 0 to leave out new objects
# treshold2_test = 0.5
repeat = 100
treshold1_test = 0.015                                                                   #maximum 10, 0 for not finiding new classes, 0 to leave out new objects
treshold2_test = 0.5


########################################################################################################################

for i in range(16):
    print("i:{}".format(i))
    if i < 4:
        objId_list = all
        objId_test_list = all
        if i == 0:
            neuron_id = random.sample(range(0, 1000), 2)
        if i == 1:
            neuron_id = random.sample(range(0, 1000), 10)
        if i == 2:
            neuron_id = random.sample(range(0, 1000), 100)
        if i == 3:
            neuron_id = all

    elif i < 8:
        objId_list = random.sample(range(1, 127), 100)
        objId_test_list = all
        if i == 4:
            neuron_id = random.sample(range(0, 1000), 2)
        if i == 5:
            neuron_id = random.sample(range(0, 1000), 10)
        if i == 6:
            neuron_id = random.sample(range(0, 1000), 100)
        if i == 7:
            neuron_id = all
    elif i < 12:
        pool_list = random.sample(range(1, 127), 15)
        objId_list = random.sample(pool_list, 10)
        objId_test_list = pool_list
        if i == 8:
            neuron_id = random.sample(range(0, 1000), 2)
        if i == 9:
            neuron_id = random.sample(range(0, 1000), 10)
        if i == 10:
            neuron_id = random.sample(range(0, 1000), 100)
        if i == 11:
            neuron_id = all
    elif i < 16:
        objId_test_list = random.sample(range(1, 127), 10)
        objId_list = objId_test_list
        if i == 12:
            neuron_id = random.sample(range(0, 1000), 2)
        if i == 13:
            neuron_id = random.sample(range(0, 1000), 10)
        if i == 14:
            neuron_id = random.sample(range(0, 1000), 100)
        if i == 15:
            neuron_id = all
    ####################################################################################################################


    # z = 1
    # pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Strategies/Test/number.pickle", "wb"))

    z = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/number.pickle", "rb"))

    z += 1

    pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Strategies/Test/number.pickle", "wb"))

    name = "/hri/storage/user/jradojev/Version1/Strategies/Test/Test{}.pickle"
    pic = "/hri/storage/user/jradojev/Version1/Strategies/Test/Test{}.png"
    name_pic1 = "/hri/storage/user/jradojev/Version1/Baseline_data/Precision{}.png"
    with open(name.format(z), 'w') as f:
        pickle.dump([objId_list, objId_test_list, neuron_id, n_neighbors, f1, k, limit_test_views, repeat, treshold, treshold1_test, treshold2_test], f)



    ########################################################################################################################
    #MAIN

    classification_rate_average = []
    precision_total_average = []

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
        if i < 4:
            objId_list = range(1, 127)
            objId_test_list = range(1, 127)
            if i == 0:
                neuron_id = random.sample(range(0, 1000), 2)
            if i == 1:
                neuron_id = random.sample(range(0, 1000), 10)
            if i == 2:
                neuron_id = random.sample(range(0, 1000), 100)
            if i == 3:
                neuron_id = range(1000)

        elif i < 8:
            objId_list = random.sample(range(1, 127), 100)
            objId_test_list = range(1, 127)
            if i == 4:
                neuron_id = random.sample(range(0, 1000), 2)
            if i == 5:
                neuron_id = random.sample(range(0, 1000), 10)
            if i == 6:
                neuron_id = random.sample(range(0, 1000), 100)
            if i == 7:
                neuron_id = range(1000)
        elif i < 12:
            pool_list = random.sample(range(1, 127), 15)
            objId_list = random.sample(pool_list, 10)
            objId_test_list = pool_list
            if i == 8:
                neuron_id = random.sample(range(0, 1000), 2)
            if i == 9:
                neuron_id = random.sample(range(0, 1000), 10)
            if i == 10:
                neuron_id = random.sample(range(0, 1000), 100)
            if i == 11:
                neuron_id = range(1000)
        elif i < 16:
            objId_test_list = random.sample(range(1, 127), 10)
            objId_list = objId_test_list
            if i == 12:
                neuron_id = random.sample(range(0, 1000), 2)
            if i == 13:
                neuron_id = random.sample(range(0, 1000), 10)
            if i == 14:
                neuron_id = random.sample(range(0, 1000), 100)
            if i == 15:
                neuron_id = range(1000)

        print ("Run: {}".format(t))
        classification_rate = []
        precision_total_1 = []
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
                confidence1, label1 = confidence_calc(test_data, training_data, training_labels, neigh, n_neighbors, treshold1_test, treshold2_test)

                if side_selector[0] == 0:
                    view_test_list = view_test_list_main[g:(g+k)]
                else:
                    view_test_list = view_test_list_main[(g-k):g][::-1]

                test_data, test_labels = get_views1([objId], view_test_list, data_location, neuron_id)
                confidence2, label2 = confidence_calc(test_data, training_data, training_labels, neigh, n_neighbors, treshold1_test, treshold2_test)

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
                        confidence2, label2 = confidence_calc(test_data, training_data, training_labels, neigh, n_neighbors,
                                                                      treshold1_test, treshold2_test)
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

            precision_total = 0
            for d in range(len(objId_test_list)):
                precision = 0
                ukupno = 0
                array1 = np.asarray(predicted_classes) == objId_test_list[d]
                for j in range(len(array1)):
                    if array1[j] == 1:
                        if test_labels_total[j] == objId_test_list[d]:
                            precision += 1
                        ukupno += 1
                if ukupno != 0:
                    precision_total += precision / ukupno
            precision_total_1.append(precision_total / len(objId_test_list))

            # classification_rate.append((np.sum(match_classes_array) + b) / float(len(match_classes_array)))  #my first method
            # print classification_rate
            # classification_rate.append(np.sum(match_classes_array) / float(len(match_classes_array)))
            # print ("Classification rate: {}".format(classification_rate))

        classification_rate_average.append(classification_rate)
        precision_total_average.append(precision_total_1)
        # print ("Classification rate average: {}".format(classification_rate_average))

    classification_rate_average = np.sum(classification_rate_average,0)/float(repeat)
    precision_total_average = np.sum(precision_total_average, 0) / float(repeat)
    # print ("Classification rate average: {}".format(classification_rate_average))


    name_class = "/hri/storage/user/jradojev/Version1/Strategies/Test/Test_Rate{}.pickle"
    with open(name_class.format(z), 'w') as f:
        pickle.dump([classification_rate_average], f)

    name_k = "/hri/storage/user/jradojev/Version1/Strategies/Test/k{}.pickle"
    with open(name_k.format(z), 'w') as f:
        pickle.dump([k_array], f)

    name_precision = "/hri/storage/user/jradojev/Version1/Baseline_data/Precision_Rate{}.pickle"
    with open(name_precision.format(z), 'w') as f:
        pickle.dump([precision_total_average], f)

    #plot
    plt.figure()
    plt.plot(number_of_training_samples_list, classification_rate_average, '-bo')
    plt.xlabel("Number of training samples per class")
    plt.ylabel("Correct classification rate")
    plt.title("Test: {} run, {} trob, {} teob, {} neuro, {} tres1, {} frames, {} neighbors".format(repeat, len(objId_list), len(objId_test_list), len(neuron_id), treshold1_test, k, n_neighbors))
    plt.ylim([0, 1])
    plt.savefig(pic.format(z))
    # plt.show()

    plt.figure()
    plt.plot(number_of_training_samples_list, precision_total_average, '-bo')
    plt.xlabel("Number of training samples per class")
    plt.ylabel("Precision")
    plt.title(
        "Test: {} run, {} trob, {} teob, {} neuro, {} tres1, {} frames, {} neighbors".format(repeat, len(objId_list),
                                                                                             len(objId_test_list),
                                                                                             len(neuron_id),
                                                                                             treshold1_test, k,
                                                                                             n_neighbors))
    plt.ylim([0, 1])
    plt.savefig(name_pic1.format(z))
    # plt.show()