#DESCRIPTION
#This is baseline strategy which doesn't take fixed number of frames for testing
#Fistly conf_test_strategy is performed and file containg how many views of the certain object system "saw" is created
#Then the same number is used for baseline, to avoid argument that performance of test is better just because more views is seen
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import random
from get_views2 import get_views2
from get_views1 import get_views1
from sklearn.neighbors import KNeighborsClassifier
from get_training_indexes import get_training_indexes
import pickle
########################################################################################################################
#PARAMETERS

total_views = 1200
# number_of_training_samples_list = [10, 50, 100, 200, 500]
number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]
training_strategy = "random"
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"
n_neighbors = 5
k = 100
repeat = 100

#########################################################################################################################

for i in [8,9,10,11,12,13,14,15]:
    print("i:{}".format(i))
    if i < 4:
        objId_list = all
        objId_test_list = all
        if i==0:
            neuron_id = random.sample(range(0, 1000), 2)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k425.pickle", "rb"))
        if i==1:
            neuron_id = random.sample(range(0, 1000), 10)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k445.pickle", "rb"))
        if i==2:
            neuron_id = random.sample(range(0, 1000), 100)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k456.pickle", "rb"))
        if i==3:
            neuron_id = all
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k480.pickle", "rb"))

    elif i < 8:
        objId_list = random.sample(range(1, 127), 100)
        objId_test_list = all
        if i == 4:
            neuron_id = random.sample(range(0, 1000), 2)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k495.pickle", "rb"))
        if i == 5:
            neuron_id = random.sample(range(0, 1000), 10)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k485.pickle", "rb"))
        if i == 6:
            neuron_id = random.sample(range(0, 1000), 100)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k562.pickle", "rb"))
        if i == 7:
            neuron_id = all
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k503.pickle", "rb"))
    elif i < 12:
        pool_list = random.sample(range(1, 127), 15)
        objId_list = random.sample(pool_list, 10)
        objId_test_list = pool_list
        if i == 8:
            neuron_id = random.sample(range(0, 1000), 2)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k584.pickle", "rb"))
        if i == 9:
            neuron_id = random.sample(range(0, 1000), 10)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k585.pickle", "rb"))
        if i == 10:
            neuron_id = random.sample(range(0, 1000), 100)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k586.pickle", "rb"))
        if i == 11:
            neuron_id = all
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k587.pickle", "rb"))
    elif i < 16:
        objId_test_list = random.sample(range(1, 127), 10)
        objId_list = objId_test_list
        if i == 12:
            neuron_id = random.sample(range(0, 1000), 2)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k644.pickle", "rb"))
        if i == 13:
            neuron_id = random.sample(range(0, 1000), 10)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k645.pickle", "rb"))
        if i == 14:
            neuron_id = random.sample(range(0, 1000), 100)
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k642.pickle", "rb"))
        if i == 15:
            neuron_id = all
            k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/k643.pickle", "rb"))
    ####################################################################################################################

    # z = 4
    # pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Baseline_data/number.pickle", "wb"))
    #
    z = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/number.pickle", "rb"))

    z += 1

    pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Baseline_data/number.pickle", "wb"))

    name = "/hri/storage/user/jradojev/Version1/Baseline_data/Bnb/Baseline{}.pickle"
    name_pic = "/hri/storage/user/jradojev/Version1/Baseline_data/Bnb/Baseline{}.png"
    name_pic1 = "/hri/storage/user/jradojev/Version1/Baseline_data/Precision{}.png"
    with open(name.format(z), 'w') as f:
        pickle.dump([objId_list, objId_test_list, neuron_id, n_neighbors, k, repeat], f)

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
        total_training_data, view_test_list_main = [], []
        print ("Run: {}".format(t))
        classification_rate = []
        precision_total_1 = []
        training_index = random.sample(range(total_views), 1)[0]
        total_training_data, view_test_list_main = get_training_indexes(training_index, k)

        # print len(total_training_data)
        # print len(view_test_list_main)

        for e in number_of_training_samples_list:
            a = random.sample(range(len(total_training_data)-e), 1)[0]
            view_training_list = total_training_data[a:(a+e)]

            #gets training data and training labels
            training_data, training_labels = get_views1(objId_list, view_training_list, data_location, neuron_id)
            neigh = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(training_data, training_labels)

            # f = range(100, 100)
            g = 100
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
                k = int(k_array[0][t][number_of_training_samples_list.index(e)][objId-1])
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

            # precision_total = 0
            # for i in range(len(objId_test_list)):
            #     precision = 0
            #     ukupno = 0
            #     array1 = np.asarray(predicted_classes_total) == objId_test_list[i]
            #     for j in range(len(array1)):
            #         if array1[j] == 1:
            #             if test_labels_total[j] == objId_test_list[i]:
            #                 precision += 1
            #             ukupno += 1
            #     if ukupno != 0:
            #         precision_total += precision / ukupno
            # precision_total_1.append(precision_total / len(objId_test_list))

        classification_rate_average.append(classification_rate)
        # precision_total_average.append(precision_total_1)
        # print ("Classification rate average: {}".format(classification_rate_average))

    classification_rate_average = np.sum(classification_rate_average, 0) / float(repeat)
    # precision_total_average = np.sum(precision_total_average, 0) / float(repeat)
    # print ("Classification rate average: {}".format(classification_rate_average))

    name_class = "/hri/storage/user/jradojev/Version1/Baseline_data/Bnb/Baseline_Rate{}.pickle"
    with open(name_class.format(z), 'w') as f:
        pickle.dump([classification_rate_average], f)

    # name_precision = "/hri/storage/user/jradojev/Version1/Baseline_data/Precision_Rate{}.pickle"
    # with open(name_precision.format(z), 'w') as f:
    #     pickle.dump([precision_total_average], f)

    plt.figure()
    plt.plot(number_of_training_samples_list, classification_rate_average, '-bo')
    plt.xlabel("Number of training samples per class")
    plt.ylabel("Correct classification rate")
    plt.title(
        "Bnb: {} r, {} tro, {} teo, {} neur, {} neigh".format(repeat, len(objId_list),
                                                                                            len(objId_test_list),
                                                                                            len(neuron_id), n_neighbors))
    plt.ylim([0, 1])
    plt.savefig(name_pic.format(z))
    # plt.show()

    # plt.figure()
    # plt.plot(number_of_training_samples_list, precision_total_average, '-bo')
    # plt.xlabel("Number of training samples per class")
    # plt.ylabel("Precision")
    # plt.title(
    #     "Bnb: {} runs, {} trobj, {}teobj, {} frames, {} neurons, {} neighbors".format(repeat, len(objId_list),
    #                                                                                        len(objId_test_list), k,
    #                                                                                        len(neuron_id), n_neighbors))
    # plt.ylim([0, 1])
    # plt.savefig(name_pic1.format(z))
    # plt.show()




