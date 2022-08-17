#DESCRIPTION
#This is baseline strategy which doesn't take fixed number of frames for testing
#Fistly conf_test_strategy is performed and file containg how many views of the certain object system "saw" is created
#Then the same number is used for baseline, to avoid argument that performance of test is better just because more views is seen
#Different than nostrategy3 is that it takes some new objects in testing
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
classification_rate_average = []
# number_of_training_samples_list = [10, 50, 100, 200, 500]
number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]
training_strategy = "random"
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"
# objId_list = random.sample(range(126), 10)
# objId_list = [6, 7, 70, 55, 81, 2, 40, 20, 85, 90]
# objId_test_list = [6, 7, 70, 55, 81, 2, 40, 21, 86, 91] #set objId_list=all for all objects
# objId_list=[6,90]
objId_list = random.sample(range(1, 126), 100)
objId_test_list = random.sample(range(1, 126), 100)
# neuron_id=[200,722]
# neuron_id = [200, 722, 300, 999]
# neuron_id = [200, 722, 300, 999, 500, 100]        #set neuron_id=all for all neurons
neuron_id = all
n_neighbors = 5
k = 50
repeat = 100

# z = 4
# pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Baseline_data/number.pickle", "wb"))
#
z = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/number.pickle", "rb"))

z += 1

pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Baseline_data/number.pickle", "wb"))

name = "/hri/storage/user/jradojev/Version1/Baseline_data/Bnb+new/Baseline{}.pickle"
name_pic = "/hri/storage/user/jradojev/Version1/Baseline_data/Bnb+new/Baseline{}.png"
with open(name.format(z), 'w') as f:
    pickle.dump([objId_list, objId_test_list, neuron_id, n_neighbors, k, repeat], f)

########################################################################################################################
#MAIN

if objId_list == all:
    objId_list = range(1, 127)

if objId_test_list == all:
    objId_test_list = range(1, 127)

if neuron_id == all:
    neuron_id = range(1000)

k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/New/k101.pickle", "rb"))

for t in range(repeat):
    objId_list = random.sample(range(1, 126), 100)
    objId_test_list = random.sample(range(1, 126), 100)
    # k = random.sample(range(5, 50, 5), 1)[0]
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
            k = int(k_array[0][t][number_of_training_samples_list.index(e)][objId-1])
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

name_class = "/hri/storage/user/jradojev/Version1/Baseline_data/Bnb+new/Baseline_Rate{}.pickle"
with open(name_class.format(z), 'w') as f:
    pickle.dump([classification_rate_average], f)

#plot
plt.figure()
plt.plot(number_of_training_samples_list, classification_rate_average, '-bo')
plt.xlabel("Number of training samples per class")
plt.ylabel("Correct classification rate")
plt.title("Baseline nb+new 100 test: Average over {} runs, {} objects, {} frames, {} neurons, {} neighbors".format(repeat, len(objId_list), k, len(neuron_id), n_neighbors, training_strategy))
plt.ylim([0, 1])
plt.savefig(name_pic.format(z))
# plt.show()



