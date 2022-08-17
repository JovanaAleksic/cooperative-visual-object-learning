import numpy as np

def get_views1(objId_list, view_list, data_location, neuron_id):
    if neuron_id == all:
        neuron_id = range(1000)
    dim = len(neuron_id)
    main = []
    training_labels = []
    # print ("ObjId_list:{}".format(objId_list))
    for objId in objId_list:
        # print ("ObjId:{}".format(objId))
        # print ("View_list:{}".format(view_list))
        for viewId in view_list:
                # print ("View Id: {}".format(viewId))
                xs = []
                filename = data_location.format(
                    objId, viewId)

                # print filename

                array = np.load(filename)['data']
                for i in range(dim):
                    xs.append(array[0][neuron_id[i]])
                main.append(xs)
        training_labels.extend([objId] * len(view_list))
    data1 = np.asarray(main)
    return data1, training_labels
