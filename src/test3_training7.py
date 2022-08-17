#DESCRIPTION:
#Testing 3 + Training 7
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
from different_views3 import different_views3
from overlap_measures_temp import overlap_measures_temp
from confidence_calc3 import confidence_calc3
########################################################################################################################

def length(s):                                                                  #number of different objects in a list
    return len(list(set(s)))

########################################################################################################################
#PARAMETERS

total_views = 1200
# number_of_training_samples_list = [10, 50, 100, 200, 500]
number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]
training_strategy = "random"
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"

n_neighbors = 5
k = 5
f1 = 4
repeat = 100
training_treshold = 0.7                                                        #MUST BE under 0.8, doesn't make sense otherwise
# treshold1 = 6
# treshold2 = 30
# treshold3=5
step = 6

# treshold1 = 0.35
# treshold2 = 0.2
# treshold3 = 0.15

# a_p = 0.015
# b_p = 0.015
# c_p = 0.3

limit_test_views = 100
treshold = 0.8
a_p = 1
per = 75
b_p = 1
c_p = 0.1
gog = 2
treshold2 = 0.7
local_h = 2
perse = 0.6

#########################################################################################################################

for i in [12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3]:
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
    # pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Strategies/Training3/number.pickle", "wb"))

    z = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training3/number.pickle", "rb"))

    z += 1

    pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Strategies/Training3/number.pickle", "wb"))

    name = "/hri/storage/user/jradojev/Version1/Strategies/Test3Training7/Test3Training7_{}.pickle"
    pic = "/hri/storage/user/jradojev/Version1/Strategies/Test3Training7/Test3Training7_{}.png"
    with open(name.format(z), 'w') as f:
        pickle.dump([objId_list, objId_test_list, neuron_id, n_neighbors, k, repeat, training_treshold, a_p, b_p, c_p, step, gog, local_h, perse, treshold, f1, limit_test_views], f)

    ########################################################################################################################
    #MAIN
    classification_rate_average = []

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
        rewardd = []
        total_training_data, view_test_list_main = [], []
        print ("Run: {}".format(t))
        classification_rate = []
        training_labels = []
        training_index = random.sample(range(total_views), 1)[0]
        total_training_data, view_test_list_main = get_training_indexes(training_index, k)

        #automatic determination of parameters

        n_neighbors_param = 6
        param_training_list = total_training_data
        q = random.sample(objId_list, 1)
        training_data_param, training_labels_param = get_views1(q, param_training_list, data_location, neuron_id)
        neigh_param = KNeighborsClassifier(n_neighbors_param, algorithm='brute').fit(training_data_param, training_labels_param)

        param_test_list = param_training_list

        test_data_param, test_labels_param = get_views1(q, param_test_list, data_location, neuron_id)
        kappa_param, gamma_param, delta_param, indices_param = overlap_measures_temp(neigh_param, n_neighbors_param, training_data_param, test_data_param)

        gamma_average = sum(gamma_param)/len(gamma_param)
        delta_average = sum(delta_param)/len(delta_param)

        # print ("Size")
        treshold_t = gamma_average * a_p
        size1 = sum(gamma_param < treshold_t) / float(len(gamma_param)) * 100
        size2 = sum(gamma_param < treshold_t) / float(len(gamma_param)) * 100
        while size1 > per:
            a_p = a_p - 0.05
            treshold_t = gamma_average * a_p
            size1 = sum(gamma_param < treshold_t)/float(len(gamma_param))*100

        # print ("a_p: {}".format(a_p))

        # while size2 < per_m:
        #     b_p = b_p + 0.05
        #     treshold_max = gamma_average*b_p
        #     size2 = sum(gamma_param < treshold_max) / float(len(gamma_param)) * 100

        # treshold_max = max(gamma_param)

        # b_p = a_p
        # print ("b_p: {}".format(b_p))

        # treshold1_name="Strategies/Training6/Tresholds/treshold1_{}.pickle"
        # treshold2_name = "Strategies/Training6/Tresholds/treshold2_{}.pickle"
        # treshold3_name = "Strategies/Training6/Tresholds/treshold3_{}.pickle"
        # with open(treshold1_name.format(z), 'w') as f:
        #     pickle.dump([a_p*gamma_average], f)
        # with open(treshold2_name.format(z), 'w') as f:
        #     pickle.dump([a_p*delta_average], f)
        # with open(treshold3_name.format(z), 'w') as f:
        #     pickle.dump([treshold_max], f)

        treshold1 = gog*gamma_average
        treshold3 = c_p*treshold1

        # print len(total_training_data)
        # print len(view_test_list_main)

        for e in number_of_training_samples_list:
            predicted_classes = []
            test_labels_total = []
            # print("Training number: {}".format(e))
            for objId in objId_list:
                # print("Object: {}".format(objId))
                view_training_list = []
                temporary_training = []
                temporary_labels = []
                main = []
                neigh1 = 0
                j = random.sample(range(1000), 1)[0]
                for o in range(1000):   #has to be checked! not good! sometimes doesn't add enough data
                    # print o
                    temporary_test_index = [int(total_training_data[(j+o*step) % 1000])]
                    # print ("Temp:{}".format(temporary_test_index))
                    temporary_test, test_labels = get_views1([objId], temporary_test_index, data_location, neuron_id)
                    different_reward = different_views3(temporary_training, temporary_test, n_neighbors, neigh1, treshold1, treshold2, treshold3)

                    if e == 20 and t == 0 and objId_list.index(objId)==0:
                        rewardd.append(different_reward)

                    if different_reward > training_treshold:
                        # view_training_list.append(temporary_test_index[0])
                        xs = []
                        filename = data_location.format(
                            objId, temporary_test_index[0])
                        array = np.load(filename)['data']
                        for l in range(len(neuron_id)):
                            xs.append(array[0][neuron_id[l]])
                        temporary_labels.extend([objId])
                        leny = len(temporary_labels)
                        main.append(xs)
                        temporary_training = np.asarray(main)
                        if 150 >= leny > n_neighbors:
                            treshold1 = (gog - gog*(leny - n_neighbors) / float(150 - n_neighbors) *(1-a_p/gog)) * gamma_average
                            treshold3 = c_p*treshold1

                    # temporary_training, temporary_labels = get_views1([objId], view_training_list, data_location, neuron_id)

                    if len(temporary_labels) == e:
                        break

                    if len(temporary_labels) <= n_neighbors and len(temporary_labels)>=2:
                        neigh1 = KNeighborsClassifier(len(temporary_labels)-1, algorithm='brute').fit(temporary_training, temporary_labels)
                    else:
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
            # # print g
            # side_selector = random.sample(range(2), 1)
            # # print side_selector

            g = random.sample(range(k, len(view_test_list_main) - k), 1)[0]
            side_selector = random.sample(range(2), 1)

            for objId in objId_test_list:
                # print ("Object: {}".format(objId))
                test_labels_total.append(objId)

                if side_selector[0] == 0:
                    view_test_list = view_test_list_main[g: (g + f1)]
                else:
                    view_test_list = view_test_list_main[(g - f1):g][::-1]

                test_data, test_labels = get_views1([objId], view_test_list, data_location, neuron_id)
                confidence1, label1 = confidence_calc3(test_data, training_data, training_labels, neigh, n_neighbors,
                                                       treshold_t, perse, local_h)

                if side_selector[0] == 0:
                    view_test_list = view_test_list_main[g:(g + k)]
                else:
                    view_test_list = view_test_list_main[(g - k):g][::-1]

                test_data, test_labels = get_views1([objId], view_test_list, data_location, neuron_id)
                confidence2, label2 = confidence_calc3(test_data, training_data, training_labels, neigh, n_neighbors,
                                                       treshold_t, perse, local_h)

                if confidence2 > treshold:
                    predicted_classes.append(label2)
                    k_array[t][number_of_training_samples_list.index(e)][objId_test_list.index(objId)] = k

                else:
                    if confidence2 < confidence1:
                        side_selector[0] = not (side_selector[0])
                    # print ("View test list elif: {}".format(len(view_test_list)))

                    while confidence2 < treshold and (len(view_test_list) < limit_test_views):
                        # print ("In while loop: {}".format(len(view_test_list)))
                        l = random.sample(range(k, len(view_test_list_main) - k), 1)[0]
                        if side_selector[0] == 0:
                            view_test_list.extend(view_test_list_main[l:l + k])
                        else:
                            view_test_list.extend(view_test_list_main[l - k:l][::-1])

                        test_data, test_labels = get_views1([objId], view_test_list, data_location, neuron_id)
                        confidence2, label2 = confidence_calc3(test_data, training_data, training_labels, neigh,
                                                               n_neighbors,
                                                               treshold_t, perse, local_h)
                    predicted_classes.append(label2)
                    k_array[t][number_of_training_samples_list.index(e)][objId_test_list.index(objId)] = len(view_test_list)

            b = 0
            # boolean array which says for every test instance is prediction equal to real label
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


                        # print predicted_classes_total
            # print test_labels_total

            #boolean array which says for every test instance is prediction equal to real label
            classification_rate.append((np.sum(match_classes_array)) / float(len(match_classes_array) - b))

        classification_rate_average.append(classification_rate)
        # print ("Classification rate average: {}".format(classification_rate_average))

    classification_rate_average = np.sum(classification_rate_average, 0) / float(repeat)
    # print ("Classification rate average: {}".format(classification_rate_average))


    name_class = "/hri/storage/user/jradojev/Version1/Strategies/Test3Training7/Test3Training7_Rate{}.pickle"
    with open(name_class.format(z), 'w') as f:
        pickle.dump([classification_rate_average], f)

    name_k = "/hri/storage/user/jradojev/Version1/Strategies/Test3Training7/k{}.pickle"
    with open(name_k.format(z), 'w') as f:
        pickle.dump([k_array], f)

    #plot
    plt.figure()
    plt.plot(number_of_training_samples_list, classification_rate_average, '-bo')
    plt.xlabel("Number of training samples per class")
    plt.ylabel("Correct classification rate")
    plt.title("TtN: {} runs, {} trobj, {} teobj, {} neuro, {} x, {} per, {} neigh, {} loh, {} perse".format(repeat, len(objId_list), len(objId_test_list), len(neuron_id), gog, per, n_neighbors_param, local_h, perse))
    plt.ylim([0, 1])
    plt.savefig(pic.format(z))
    # plt.show()



