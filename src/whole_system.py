#DESCRIPTION:
#WHOLE system in interaction setting
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
from phase_training import phase_training
########################################################################################################################
#PARAMETERS

total_views = 1200
# classification_rate_average = []
# number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]
data_location = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz"
objId_list = random.sample(range(1, 127), 38)
neuron_id = all
n_neighbors = 10
k = 10
f1 = 2
repeat = 10
training_treshold = 0.7
step = 1

limit_test_views = 101
treshold = 0.55
a_p = 1
per = 60
b_p = 1
c_p = 0.1
gog = 2
treshold2 = 0.7
local_h1 = 7
local_h2 = 2
perse = 0.7
see = 30

# z = 1
# pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Strategies/Whole/aanumber.pickle", "wb"))


########################################################################################################################
#MAIN

if neuron_id == all:
    neuron_id = range(1000)



for t in range(repeat):
    z = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Whole/aanumber.pickle", "rb"))

    z += 1

    pickle.dump(z, open("/hri/storage/user/jradojev/Version1/Strategies/Whole/aanumber.pickle", "wb"))

    name = "/hri/storage/user/jradojev/Version1/Strategies/Whole/w_{}.pickle"
    pic = "/hri/storage/user/jradojev/Version1/Strategies/Whole/confidence_{}.png"
    pic1 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/reward_{}.png"
    pic2 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/success_{}.png"
    pic3 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/comb_{}.png"
    pic4 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/counter_{}.png"
    pic5 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/labels_{}.png"
    with open(name.format(z), 'w') as f:
        pickle.dump(
            [objId_list, neuron_id, n_neighbors, k, repeat, training_treshold, a_p, b_p, c_p, step, gog, local_h1,
             local_h2, perse, treshold, f1, limit_test_views], f)

    counter = []
    new_list=[]
    main_labels_change=[]
    true_labels=[]
    count = 0
    lenica = 0
    added = []
    neigh = 0
    tr_reward = []
    success = []
    seen_objects = []
    training_data = []
    main_confidence = []
    training_labels = []
    predicted_classes = []
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

    print ("Size")
    treshold_t = gamma_average * a_p
    size1 = sum(gamma_param < treshold_t) / float(len(gamma_param)) * 100
    size2 = sum(gamma_param < treshold_t) / float(len(gamma_param)) * 100
    while size1 > per:
        a_p = a_p - 0.05
        treshold_t = gamma_average * a_p
        size1 = sum(gamma_param < treshold_t)/float(len(gamma_param))*100

    print ("a_p: {}".format(a_p))

    treshold1 = gog*gamma_average
    treshold3 = c_p*treshold1

    # print ("treshold1: {}".format(treshold1))
    # print ("treshold3: {}".format(treshold3))

    while lenica < see:
        counter_point = 0
        labels_change = []
        count += 1
        confidence = []
        local_h = local_h2 + (local_h1-local_h2)/(lenica*0.1+1)
        print ("Local_h: {}".format(local_h))
        print ("Object nb: {}".format(len(list(set(seen_objects)))))
        new_object = 0

        pick_object = random.sample(objId_list, 1)[0]
        true_labels.append(pick_object)
        print ("Pick object: {}".format(pick_object))

        if pick_object not in list(set(training_labels)):
            new_object = 1
            new_list.append(1)
        else:
            new_list.append(0)

        training_index = random.sample(range(total_views), 1)[0]
        total_training_data, view_test_list_main = get_training_indexes(training_index, k)

        j = random.sample(range(10, 190), 1)[0]
        side_selector = random.sample(range(2), 1)
        view_test_list = [view_test_list_main[j]]

        # print("Len:{}".format(len(training_data)))
        # print("Len labels: {}".format(len(training_labels)))


        for i in range(1, k):
            if side_selector[0] == 0:
                view_test_list.extend(view_test_list_main[j+i:j+i+1])
            else:
                view_test_list.extend(view_test_list_main[j-i - 1:j-i])

            temporary_test, test_labels = get_views1([pick_object], view_test_list, data_location, neuron_id)
            confidence1, label1 = confidence_calc3(temporary_test, training_data, training_labels, neigh, n_neighbors, treshold3, perse, local_h)
            labels_change.append(label1)
            confidence.append(confidence1)
            # counter_point += len(view_test_list)

        main_confidence.append(confidence)

        if confidence1 > treshold:
            predicted_classes.append(label1)
            counter.append(len(view_test_list))
            main_labels_change.append(labels_change)


            if label1 != pick_object:
                op, neigh, training_reward, labels_tr, add = phase_training(neigh, training_data, training_labels, n_neighbors, treshold1, treshold2, treshold3, training_treshold, data_location, neuron_id, pick_object, gog, gamma_average, a_p, c_p)
                tr_reward.append(training_reward)
                added.append(add)
                training_labels = labels_tr
                training_data = op

                if label1 == 0 and new_object == 0:
                    success.append(-0.5)
                if label1 != 0 and new_object == 0:
                    success.append(-1)
                if label1 != 0 and new_object == 1:
                    success.append(-0.75)
                if new_object == 1 and label1 == 0:
                    success.append(0.5)

            elif label1 == pick_object:
                success.append(1)
                added.append(0)



        else:
            if confidence1 < confidence[-(k-f1+1)]:
                side_selector[0] = not (side_selector[0])

            while confidence1 < treshold and (len(view_test_list) < limit_test_views):
                l = random.sample(range(k, len(view_test_list_main) - k), 1)[0]   #ovde nesto nije ok, umesto k treba da bude nes drugo

                if side_selector[0] == 0:
                    view_test_list.extend(view_test_list_main[l:l + 1])   #dal svaki put bira random, mozda treba samo ponekad
                else:
                    view_test_list.extend(view_test_list_main[l - 1:l][::-1])

                temporary_test, test_labels = get_views1([pick_object], view_test_list, data_location, neuron_id)
                confidence1, label1 = confidence_calc3(temporary_test, training_data, training_labels, neigh,  n_neighbors,treshold3, perse, local_h)
                labels_change.append(label1)
                # counter_point += len(temporary_test)
                confidence.append(confidence1)

            predicted_classes.append(label1)
            counter.append(len(view_test_list))
            main_labels_change.append(labels_change)

            if label1 != pick_object:
                op, neigh, training_reward, labels_tr, add = phase_training(neigh, training_data, training_labels, n_neighbors, treshold1, treshold2, treshold3, training_treshold, data_location, neuron_id, pick_object, gog, gamma_average, a_p, c_p)
                tr_reward.append(training_reward)
                added.append(add)
                training_labels = labels_tr
                training_data = op

                if label1 == 0 and new_object == 0:
                    success.append(-0.5)
                if label1 != 0 and new_object == 0:
                    success.append(-1)
                if label1!=0 and new_object == 1:
                    success.append(-1)
                if new_object == 1 and label1 == 0:
                    success.append(0.5)


            elif label1 == pick_object:
                go_training = random.sample(range(2), 1) #training optionaly
                if go_training == 1:
                    op, neigh, training_reward, labels_tr, add = phase_training(neigh, training_data, training_labels, n_neighbors, treshold1, treshold2, treshold3, training_treshold, data_location, neuron_id, pick_object, gog, gamma_average, a_p, c_p)
                    tr_reward.append(training_reward)
                    added.append(add)
                    training_labels = labels_tr
                    training_data = op
                    success.append(1)
                else:
                    added.append(0)
                    success.append(1)


        seen_objects.append(pick_object)
        lenica=len(list(set(seen_objects)))

    # name_class = "/hri/storage/user/jradojev/Version1/Strategies/Whole/confidence_rate{}.pickle"
    # with open(name_class.format(z), 'w') as f:
    #     pickle.dump([main_confidence], f)
    #
    # name_class1 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/tr_reward_rate{}.pickle"
    # with open(name_class1.format(z), 'w') as f:
    #     pickle.dump([tr_reward], f)
    #
    # name_class2 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/success_rate{}.pickle"
    # with open(name_class2.format(z), 'w') as f:
    #     pickle.dump([success], f)
    #
    # name_class3 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/added_rate{}.pickle"
    # with open(name_class3.format(z), 'w') as f:
    #     pickle.dump([added], f)

    name_class4 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/counter{}.pickle"
    with open(name_class4.format(z), 'w') as f:
        pickle.dump([counter], f)

    name_class5 = "/hri/storage/user/jradojev/Version1/Strategies/Whole/labels_change{}.pickle"
    with open(name_class5.format(z), 'w') as f:
        pickle.dump([main_labels_change], f)
#
    # ko = 0
    # nm = 0
    # max_nm=0
    # plt.figure(figsize=(20, 6))
    # for h in range(count):
    #     if len(main_confidence[h]) > max_nm:
    #         max_nm=len(main_confidence[h])
    #         specconf = h
    #     if h==0:
    #         plt.plot(range(nm, nm + len(main_confidence[h])), main_confidence[h], '-b', label='Testing confidence')
    #         nm += len(main_confidence[h])
    #         if added[h] != 0:
    #             plt.plot(range(nm, nm + len(tr_reward[ko])), tr_reward[ko], '-r', label='Training score')
    #             nm += len(tr_reward[ko])
    #             ko += 1
    #     else :
    #         plt.plot(range(nm, nm + len(main_confidence[h])), main_confidence[h], '-b')
    #         nm += len(main_confidence[h])
    #         if added[h] != 0:
    #             plt.plot(range(nm, nm + len(tr_reward[ko])), tr_reward[ko], '-r')
    #             nm += len(tr_reward[ko])
    #             ko += 1
    # plt.xlabel("Incoming views" , fontsize=20)
    # plt.ylabel("Testing confidence / Training score", fontsize=20)
    # # plt.title("Test - Training Combination")
    # plt.legend(loc=1, borderpad=0.9, labelspacing=1, prop={'size': 20})
    # # plt.ylim([0, max(tr_reward)+0.2])
    # plt.savefig(pic3.format(z))
    #
    #
    # tr = sum([1 for x in added if x > 0])
    # print("Sum of added training: {}".format(tr))
    # final = []
    # ko = 0
    # for h in range(count):
    #     final.append(main_confidence[h])
    #     if added[h] != 0:
    #         final.append(tr_reward[ko])
    #         ko += 1
    # main_confidence1 = []
    #
    # for h in range(count):
    #     main_confidence1.extend(main_confidence[h])
    #
    # plt.figure(figsize=(15, 6))
    # # ax = fig1.gca()
    # # ax.axis('tight')
    # # fig1.tight_layout()
    # plt.plot(main_confidence[specconf], '-b')
    # plt.xlabel("Testing samples", fontsize=20)
    # plt.ylabel("Confidence measure", fontsize=20)
    # plt.grid()
    # plt.savefig(pic.format(z))
    #
    # reward1 = []
    # max_tr=0
    # for h in range(len(tr_reward)):
    #     reward1.extend(tr_reward[h])
    #     if len(tr_reward[h]) > max_tr:
    #         max_tr=len(tr_reward[h])
    #         spectr=h
    #
    # plt.figure(figsize=(15, 6))
    # # plt.figure(figsize=(20, 5))
    # # ax = fig.gca()
    # # ax.axis('tight')
    # # fig.tight_layout()
    # plt.plot(tr_reward[spectr], '-b')
    # plt.xlabel("Training samples", fontsize=20)
    # plt.ylabel("Training score", fontsize=20)
    # plt.grid()
    # plt.savefig(pic1.format(z))
    # #
    # plt.figure()
    # # ax = fig.gca()
    # # ax.axis('tight')
    # # fig.tight_layout()
    #
    # plt.plot(range(len(success)), success, 'b-')
    # cor=[]
    # cor_i=[]
    # cor_new=[]
    # cor_new_i=[]
    # wr=[]
    # wr_i=[]
    # wr_new=[]
    # wr_new_i=[]
    # wj=[]
    # wj_i=[]
    # for i in range(len(success)):
    #     if success[i] == 1:
    #         cor.append(1)
    #         cor_i.append(i)
    #     if success[i] == 0.5:
    #         cor_new.append(0.5)
    #         cor_new_i.append(i)
    #     if success[i]==-0.5:
    #         wr_new.append(-0.5)
    #         wr_new_i.append(i)
    #     if success[i]==-1:
    #         wr.append(-1)
    #         wr_i.append(i)
    #     if success[i]==-0.75:
    #         wj.append(-0.75)
    #         wj_i.append(i)
    #
    #
    # plt.plot(cor_i, cor, 'go', color='#32cd32', markersize=12, label='Old object predicted correctly')
    # plt.plot(cor_new_i, cor_new, 's', color='#0000ff',markersize=12, label='New object predicted as new object')
    # plt.plot(wr_new_i, wr_new, '^', color='#ffff00', markersize=12, label='Old object predicted as new object')
    # plt.plot(wr_i, wr, 'rD', markersize=12, label='Old object predicted as wrong old object')
    # if len(wj)!=0:
    #     plt.plot(wj_i, wj, '*', color='#ff69b4', markersize=12, label='New object predicted as old object')
    # plt.xlabel("Objects", fontsize=20)
    # plt.ylabel("Success", fontsize=20)
    # # plt.title("Classification success")
    # plt.ylim([-1.5, 2])
    # plt.xticks(range(len(success)), fontsize=15)
    # plt.yticks([-1.0, -0.5, 0.5, 1.0], fontsize=15)
    # plt.savefig(pic2.format(z))
    # plt.legend(loc=1, borderpad=0.9, numpoints=1, labelspacing=1, prop={'size': 20})
    # plt.grid()


    #
    # plt.figure()
    # # ax = fig.gca()
    # # ax.axis('tight')
    # # fig.tight_layout()
    # plt.plot(range(len(added)), added, '-ro')
    # plt.xlabel("Objects")
    # plt.ylabel("Number of accepted samples")
    # # plt.title("Added training samples into memory")
    # plt.savefig(pic3.format(z))


    # conf_graph = main_confidence1[-1000::]
    # plt.figure(figsize=(20, 5))
    # plt.plot(range(len(conf_graph)), conf_graph, '-b')
    # plt.xlabel("Testing samples")
    # plt.ylabel("Confidence measure")
    # plt.grid()
    # plt.show()


    # if len(main_labels_change) == len(true_labels):
    #     print("Size match: {}".format(len(main_labels_change[5])))

    plt.figure(figsize=(20, 6))
    plt.plot(range(len(counter)), counter, '-k')
    plt.plot(range(len(counter)), counter, 'bo', markersize=7, label='Number of views')
    plt.title('0.55 confidence')
    plt.xlabel("Testing objects", fontsize=20)
    plt.xticks(range(1, count, 1))
    plt.ylabel("Testing views seen per object", fontsize=20)
    # plt.legend(loc=2, borderpad=0.9, numpoints=1, labelspacing=1,markerscale=1, prop={'size': 20})
    plt.yticks(range(0,110,10))
    plt.xlim([1, count-1])
    plt.ylim([0,110])
    plt.grid()
    plt.savefig(pic4.format(z))

    # max_lab = 0
    # for h in range(count):
    #     if len(main_labels_change[h]) > max_lab:
    #         max_lab = len(main_labels_change[h])
    #         speclab = h
    #
    #
    # max_change_lab = 0
    # for h in range(count):
    #     if len(list(set(main_labels_change[h]))) > max_change_lab:
    #         max_lab = len(list(set(main_labels_change[h])))
    #         specchange = h
    #
    #     if len(list(set(main_labels_change[h]))) == max_change_lab:
    #         if len(main_labels_change[h]) > len(main_labels_change(max_change_lab)):
    #             max_lab = len(list(set(main_labels_change[h])))
    #             specchange = h
    #
    # true_l=true_labels[speclab]

    # plt.figure(figsize=(20, 5))
    # plt.plot(range(len(main_labels_change[speclab])), main_labels_change[speclab], '-k')
    # plt.plot(range(len(main_labels_change[speclab])), main_labels_change[speclab], 'bo', markersize=7)
    # plt.title("Max: True label: {}".format(true_l))
    # plt.xlabel("Testing views", fontsize=20)
    # # plt.xticks(range(1, count, 1))
    # plt.ylabel("Predicted labels", fontsize=20)
    # # plt.legend(loc=2, borderpad=0.9, numpoints=1, labelspacing=1,markerscale=1, prop={'size': 20})
    # plt.yticks(list(set(main_labels_change[speclab])))
    # # plt.xlim([1, count - 1])
    # # plt.ylim([0, 110])
    # plt.grid()
    # plt.savefig(pic5.format(z))

    # f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 12))  # 20,7 for three cases
    # ax1.plot(range(1,len(main_labels_change[specchange])+1), main_labels_change[specchange], '-k')
    # ax1.plot(range(1,len(main_labels_change[specchange])+1), main_labels_change[specchange], 'ko', markersize=5)
    # ax1.set_title("Change: True label: {} Is it new: {}".format(true_l, new_list[specchange]))
    # ax2.set_xlabel("Testing views", fontsize=20)
    # ax1.set_ylabel("Predicted labels", fontsize=20)
    # ax1.set_yticks(list(set(main_labels_change[specchange])))
    # ax2.plot(range(1,len(main_confidence[specchange])+1), main_confidence[specchange], '-b')
    # ax2.plot(range(1,len(main_confidence[specchange])+1), main_confidence[specchange], 'bo', markersize=5)
    # ax2.set_ylabel("Testing confidence",fontsize=20)
    # ax2.grid()
    # ax1.grid()
    # ax2.set_ylim([0, 1.1])
    # ax2.set_yticks([0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # ax1.set_ylim([-1, max(main_labels_change[specchange])+1])
    # f.savefig(pic5.format(z))



    # plt.figure(figsize=(20, 5))
    # plt.plot(range(len(main_labels_change[specchange])), main_labels_change[specchange], '-k')
    # plt.plot(range(len(main_labels_change[specchange])), main_labels_change[specchange], 'bo', markersize=7)
    # plt.title("Change: True label: {} Is new: {}".format(true_l, new_list[specchange]))
    # plt.xlabel("Testing views", fontsize=20)
    # plt.xticks(range(1, count, 1))
    # plt.ylabel("Predicted labels", fontsize=20)
    # plt.legend(loc=2, borderpad=0.9, numpoints=1, labelspacing=1,markerscale=1, prop={'size': 20})
    # plt.yticks(list(set(main_labels_change[specchange])))
    # plt.xlim([1, count - 1])
    # plt.ylim([0, 110])
    # plt.grid()
    # plt.savefig(pic5.format(z))

    plt.show()



