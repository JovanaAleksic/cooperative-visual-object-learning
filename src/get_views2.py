import numpy as np
import random

def get_views2(objId_list, view_test_list_main, k_array, data_location, neuron_id, g, t, number_of_training_samples_list, e):
    dim = len(neuron_id)
    main = []
    training_labels = []
    side_selector = random.sample(range(2), 1)
    # print side_selector
    # print ("G: {}".format(g))
    # print ("ObjId_list:{}".format(objId_list))
    for objId in objId_list:
        # print k_array.shape
        k = int(k_array[0][t][number_of_training_samples_list.index(e)][objId_list.index(objId)-1])
        # print ("k:{}".format(k))
        if side_selector[0] == 0:
            view_list = view_test_list_main[g:(g + k)]
        else:
            view_list = view_test_list_main[(g - k + 1):(g + 1)]


        # print ("Number of views: {}".format(len(view_list)))
        # print ("ObjId:{}".format(objId))
        # print ("View_list equal to k:{}".format(len(view_list)==k))
        for viewId in view_list:
                # print ("View Id: {}".format(viewId))
                xs = []
                filename = data_location.format(
                    objId, viewId)
                array = np.load(filename)['data']
                for i in range(dim):
                    xs.append(array[0][neuron_id[i]])
                main.append(xs)
        # print ("ObjId: {}".format(objId))
        # print ("Len main: {}".format(len(main)))
        training_labels.extend([objId] * len(view_list))
    data1 = np.asarray(main)
    # print ("Data shape: {}".format(data1.shape))
    return data1, training_labels
