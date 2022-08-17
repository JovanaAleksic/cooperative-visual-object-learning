import numpy as np

def get_views(objId, view_list, x, y, neuron_id):
    for viewId in view_list:
                filename = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz".format(
                    objId, viewId)
                array = np.load(filename)['data']
                x.append(array[0][neuron_id[0]])
                y.append(array[0][neuron_id[1]])

    return x, y