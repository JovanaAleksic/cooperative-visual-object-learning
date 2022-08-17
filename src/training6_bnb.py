#DESCRIPTION:
#Same training strategy as training_strategy5, just with change of tresholds with time
#Take all views of a first object and determine according to that, depending on the number of training samples acquired, tresholds lower down over time
#Testing as in nostrategy2
########################################################################################################################
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import random
from get_views1 import get_views1
from sklearn.neighbors import KNeighborsClassifier
from get_training_indexes import get_training_indexes
import pickle
from different_views2 import different_views2
from overlap_measures_temp import overlap_measures_temp
from get_views2 import get_views2
########################################################################################################################

def length(s):                                                                  #number of different objects in a list
    return len(list(set(s)))

########################################################################################################################
#PARAMETERS

total_views = 1200
# number_of_training_samples_list = [200]
number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]
training_strategy = "random"
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"

n_neighbors = 5
k = 5
repeat = 100
training_treshold = 0.7                                                        #MUST BE under 0.8, doesn't make sense otherwise
# treshold1 = 6
# treshold2 = 30
# treshold3=5
step = 1

# treshold1 = 0.35
# treshold2 = 0.2
# treshold3 = 0.15

# a_p = 0.015
# b_p = 0.015
# c_p = 0.3

a_p = 1
per = 80
# per_m = 90
b_p = 1
c_p = 0.1
gog = 2

########################################################################################################################
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

    # z = 1
    # pickle.dump(z, open("Strategies/Training3/number.pickle", "wb"))

    z = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training3/number.pickle", "rb"))

    z += 1

    pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Strategies/Training3/number.pickle", "wb"))

    name = "/hri/storage/user/jradojev/Version1/Strategies/Training6+bnb/Training{}.pickle"
    pic = "/hri/storage/user/jradojev/Version1/Strategies/Training6+bnb/Training{}.png"
    with open(name.format(z), 'w') as f:
        pickle.dump([objId_list, objId_test_list, neuron_id, n_neighbors, k, repeat, training_treshold, a_p, b_p, c_p, step, gog], f)

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
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 1:
                neuron_id = random.sample(range(0, 1000), 10)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 2:
                neuron_id = random.sample(range(0, 1000), 100)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 3:
                neuron_id = range(1000)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))

        elif i < 8:
            objId_list = random.sample(range(1, 127), 100)
            objId_test_list = range(1, 127)
            if i == 4:
                neuron_id = random.sample(range(0, 1000), 2)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 5:
                neuron_id = random.sample(range(0, 1000), 10)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 6:
                neuron_id = random.sample(range(0, 1000), 100)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 7:
                neuron_id = range(1000)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
        elif i < 12:
            pool_list = random.sample(range(1, 127), 15)
            objId_list = random.sample(pool_list, 10)
            objId_test_list = pool_list
            if i == 8:
                neuron_id = random.sample(range(0, 1000), 2)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 9:
                neuron_id = random.sample(range(0, 1000), 10)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 10:
                neuron_id = random.sample(range(0, 1000), 100)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 11:
                neuron_id = range(1000)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
        elif i < 16:
            objId_test_list = random.sample(range(1, 127), 10)
            objId_list = objId_test_list
            if i == 12:
                neuron_id = random.sample(range(0, 1000), 2)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 13:
                neuron_id = random.sample(range(0, 1000), 10)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 14:
                neuron_id = random.sample(range(0, 1000), 100)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
            if i == 15:
                neuron_id = range(1000)
                k_array = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/k97.pickle", "rb"))
        tre = []
        rewardd = []
        lenyy=[]
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


        gamma_average = sum(gamma_param) / float(len(gamma_param))
        delta_average = sum(delta_param) / float(len(delta_param))

        a_p = 1

        print ("Size")
        treshold_t = gamma_average * a_p
        size1 = sum(gamma_param < treshold_t) / float(len(gamma_param)) * 100
        # size2 = sum(gamma_param < treshold_t) / float(len(gamma_param)) * 100
        while size1 < per:
            a_p = a_p + 0.01
            treshold_t = gamma_average * a_p
            size1 = sum(gamma_param < treshold_t)/float(len(gamma_param))*100

        print ("a_p: {}".format(a_p))

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
        tre.append(treshold1)
        print ("Treshold1_first: {}".format(treshold1))
        print ("Treshold1_end: {}".format(a_p*gamma_average))
        treshold2 = b_p*delta_average
        treshold3 = c_p*treshold1

        # print len(total_training_data)
        # print len(view_test_list_main)

        for e in number_of_training_samples_list:
            # print("Training number: {}".format(e))
            for objId in objId_list:
                list_of_taken_views=[]
                # print("Object: {}".format(objId))
                view_training_list = []
                temporary_training = []
                temporary_labels = []
                s = 0
                s2 = 0
                s3 = 0
                main = []
                neigh1 = 0
                j = random.sample(range(1000), 1)[0]
                for i in range(4000):   #has to be checked! not good! sometimes doesn't add enough data
                    # print i
                    temporary_test_index = [int(total_training_data[(j+i*step) % 1000])]
                    # print ("Temp:{}".format(temporary_test_index))
                    temporary_test, test_labels = get_views1([objId], temporary_test_index, data_location, neuron_id)
                    different_reward = different_views2(temporary_training, temporary_test, n_neighbors, neigh1, treshold1, treshold2, treshold3)

                    if e == number_of_training_samples_list[0] and t == 0 and objId_list.index(objId) == 0:
                        rewardd.append(different_reward)

                    if different_reward > training_treshold and temporary_test_index not in list_of_taken_views:
                        list_of_taken_views.append(temporary_test_index)
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
                        if 150 > leny > n_neighbors and i < 1000:
                            treshold1 = (gog - gog*(leny - n_neighbors) / float(150 - n_neighbors) * (1-(a_p/float(gog)))) * gamma_average
                            if e == number_of_training_samples_list[0] and t == 0 and objId_list.index(objId) == 0:
                                tre.append(treshold1)
                            # treshold2 = (gog - gog*(leny - n_neighbors) / float(100 - n_neighbors) * (
                            # 1 - a_p/gog)) * delta_average
                            treshold3 = c_p*treshold1
                            # if treshold1 == 0.0:
                            #     print ("Whaaaaaat 1")

                    if i > 999 and leny <= (e/2):
                        # if treshold1==0.0 and s==0:
                        #     print ("Whaaaaaat")
                        # if treshold1 == 0.0 and s == 1:
                        #     print ("Whaaaaaat 2")
                        if s==0:
                            # print ("Tres: {}".format(treshold1))
                            treshold1 = treshold1*0.9
                            # print ("Tres_now: {}".format(treshold1))
                            treshold3 = c_p * treshold1
                            s=1
                            if e == number_of_training_samples_list[0] and t == 0:
                                tre.append(treshold1)
                        else:
                            # print ("Tres s=1: {}".format(treshold1))
                            treshold1 = treshold1*0.999
                            # print ("Tres_now s=1: {}".format(treshold1))
                            treshold3 = c_p * treshold1
                            if e == number_of_training_samples_list[0] and t == 0:
                                tre.append(treshold1)
                    if i > 1999 and leny <= e*0.75:
                        if s2==0:
                            treshold1 = treshold1*0.8
                            treshold3 = c_p * treshold1
                            s2 = 1
                            if e == number_of_training_samples_list[0] and t == 0:
                                tre.append(treshold1)
                        else:
                            treshold1 = treshold1*0.99
                            treshold3 = c_p * treshold1
                            if e == number_of_training_samples_list[0] and t == 0:
                                tre.append(treshold1)

                    if i > 2999 and leny < e*0.9:
                        if s3==0:
                            treshold1=treshold1*0.6
                            if e == number_of_training_samples_list[0] and t == 0:
                                tre.append(treshold1)
                        else:
                            treshold1 = treshold1 * 0.98
                            if e == number_of_training_samples_list[0] and t == 0:
                                tre.append(treshold1)



                    # temporary_training, temporary_labels = get_views1([objId], view_training_list, data_location, neuron_id)

                    if len(temporary_labels) == e:
                        break

                    if len(temporary_labels) <= n_neighbors and len(temporary_labels)>=2:
                        neigh1 = KNeighborsClassifier(len(temporary_labels)-1, algorithm='brute').fit(temporary_training, temporary_labels)
                    else:
                        neigh1 = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(temporary_training, temporary_labels)

                # print ("Treshold1: {}".format(treshold1))
                print len(temporary_labels)
                # lenyy.append(len(temporary_labels))

                if objId_list.index(objId) != 0:
                    training_data = np.append(training_data, temporary_training, 0)
                    training_labels.extend(temporary_labels)
                else:
                    training_data = temporary_training
                    training_labels = temporary_labels

                # print("Added {} training samples".format(len(temporary_labels)))  #

            neigh = KNeighborsClassifier(n_neighbors, algorithm='brute').fit(training_data, training_labels)

            f = range(50, 150)
            g = random.sample(f, 1)[0]
            # print view_test_list
            # gets test data and real test labels
            test_data, test_labels = get_views2(objId_test_list, view_test_list_main, k_array, data_location, neuron_id, g,
                                                t, number_of_training_samples_list, e)
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
                k = int(k_array[0][t][number_of_training_samples_list.index(e)][objId_test_list.index(objId) - 1])
                # print objId
                predicted_classes_total.append(max(predicted_classes[0:k], key=predicted_classes[0:k].count))
                test_labels_total.append(max(test_labels[0:k], key=test_labels[0:k].count))
                predicted_classes = predicted_classes[k::]
                test_labels = test_labels[k::]
                # print len(predicted_classes)

            # print predicted_classes_total
            # print test_labels_total

            # boolean array which says for every test instance is prediction equal to real label
            match_classes_array = np.asarray(predicted_classes_total) == np.asarray(test_labels_total)
            classification_rate.append(np.sum(match_classes_array) / float(len(match_classes_array)))

        classification_rate_average.append(classification_rate)
        # print ("Classification rate average: {}".format(classification_rate_average))

    classification_rate_average = np.sum(classification_rate_average, 0) / float(repeat)
    # print ("Classification rate average: {}".format(classification_rate_average))



    name_class = "/hri/storage/user/jradojev/Version1/Strategies/Training6+bnb/Training_Rate{}.pickle"
    with open(name_class.format(z), 'w') as f:
        pickle.dump([classification_rate_average], f)

    #plot
    plt.figure()
    plt.plot(number_of_training_samples_list, classification_rate_average, '-bo')
    plt.xlabel("Number of training samples per class")
    plt.ylabel("Correct classification rate")
    plt.title("Tr: {} runs, {} trobj, {} teobj, {} neuro, {} x, {} per, {} neigh".format(repeat, len(objId_list), len(objId_test_list), len(neuron_id), gog, per,  n_neighbors_param))
    plt.ylim([0, 1])
    plt.savefig(pic.format(z))
    # plt.show()

    plt.figure()
    plt.plot(range(1, len(rewardd)+1), rewardd, '-bo')
    plt.title("{} tr, {} per, {} gog ".format(number_of_training_samples_list[0], per, gog))
    plt.xlabel("Number of training samples")
    plt.ylabel("Reward")
    plt.show()

    plt.figure()
    plt.plot(range(1, len(tre)+1), tre, '-ro')
    plt.title("Treshold: {} tr, {} per, {} gog ".format(number_of_training_samples_list[0], per, gog))
    plt.xlabel("Number of training samples")
    plt.ylabel("Treshold value")
    # plt.show()

    # plt.figure()
    # plt.plot(range(1, len(lenyy)+1), lenyy, '-go')
    # plt.title("{} tr, {} per, {} gog ".format(number_of_training_samples_list[0], per, gog))
    # plt.xlabel("Number of training samples")
    # plt.ylabel("Added data")
    # plt.show()






