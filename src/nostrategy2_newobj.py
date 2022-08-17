#DESCRIPTION
#This is basic strategy-nostrategy2, it takes k consequtive frames for testing
#Different than nostrategy2 is that it takes some new objects in testing
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import random
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
k = 5
repeat = 100

#########################################################################################################################

for i in range(16):
    print("i:{}".format(i))
    if i < 4:
        objId_list = all
        objId_test_list = all
        if i==0:
            neuron_id = random.sample(range(0, 1000), 2)
        if i==1:
            neuron_id = random.sample(range(0, 1000), 10)
        if i==2:
            neuron_id = random.sample(range(0, 1000), 100)
        if i==3:
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


    # z = 4
    # pickle.dump(z, open("Baseline_data/number.pickle", "wb"))
    #
    z = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/number.pickle", "rb"))

    z += 1

    pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Baseline_data/number.pickle", "wb"))

    name = "/hri/storage/user/jradojev/Version1/Baseline_data/New/Baseline{}.pickle"
    name_pic = "/hri/storage/user/jradojev/Version1/Baseline_data/New/Baseline{}.png"
    with open(name.format(z), 'w') as f:
        pickle.dump([objId_list, objId_test_list, neuron_id, n_neighbors, k, repeat], f)

    ########################################################################################################################
    #MAIN
    classification_rate_average = []
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
        training_index = random.sample(range(total_views), 1)[0]
        total_training_data, view_test_list_main = get_training_indexes(training_index, k)

        # print len(total_training_data)
        # print len(view_test_list_main)

        for e in number_of_training_samples_list:

            if training_strategy == "random":
                a = random.sample(range(len(total_training_data)-e), 1)[0]
                view_training_list = total_training_data[a:(a+e)]
                # print view_training_list
            else:
                print("Unknown training strategy. Pick a correct training strategy!")


            #gets training data and training labels
            training_data, training_labels = get_views1(objId_list, view_training_list, data_location, neuron_id)
            neigh = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(training_data, training_labels)

            g = random.sample(range(k, len(view_test_list_main)-k), 1)[0]
            # print g
            side_selector = random.sample(range(2), 1)
            # print side_selector

            if side_selector[0] == 0:
                view_test_list = view_test_list_main[g:(g+k)]
            else:
                view_test_list = view_test_list_main[(g-k+1):(g+1)]

            # print view_test_list
            #gets test data and real test labels
            test_data, test_labels = get_views1(objId_test_list, view_test_list, data_location, neuron_id)
            predicted_classes = neigh.predict(test_data)

            predicted_classes = predicted_classes.tolist()
            predicted_classes_total = []
            test_labels_total = []

            for i in range(len(objId_list)):
                predicted_classes_total.append(max(predicted_classes[i*k:(i+1)*k], key=predicted_classes[i*k:(i+1)*k].count))
                test_labels_total.append(max(test_labels[i*k:(i+1)*k], key=test_labels[i*k:(i+1)*k].count))

            # print predicted_classes_total
            # print test_labels_total

            #boolean array which says for every test instance is prediction equal to real label
            match_classes_array = np.asarray(predicted_classes_total) == np.asarray(test_labels_total)
            classification_rate.append(np.sum(match_classes_array)/float(len(match_classes_array)))

        classification_rate_average.append(classification_rate)
        # print ("Classification rate average: {}".format(classification_rate_average))

    classification_rate_average = np.sum(classification_rate_average, 0) / float(repeat)
    # print ("Classification rate average: {}".format(classification_rate_average))

    name_class = "/hri/storage/user/jradojev/Version1/Baseline_data/New/Baseline_Rate{}.pickle"
    with open(name_class.format(z), 'w') as f:
        pickle.dump([classification_rate_average], f)

    #plot
    plt.figure()
    plt.plot(number_of_training_samples_list, classification_rate_average, '-bo')
    plt.xlabel("Number of training samples per class")
    plt.ylabel("Correct classification rate")
    plt.title("Baseline: Average over {} runs, {} objects, {} frames, {} neurons, {} neighbors".format(repeat, len(objId_list), k, len(neuron_id), n_neighbors, training_strategy))
    plt.ylim([0, 1])
    plt.savefig(name_pic.format(z))
    # plt.show()



