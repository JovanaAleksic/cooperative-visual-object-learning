import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from scipy.spatial import distance


# data = np.load('obj6__0.npz')
# array = data['data']
# print(array[0])

view_number = 1200

objId = 6

data = []
# list1 = [0, 1, 2, 3, 4, 107, 137, 138, 139, 140, 141, 142, 143, 320, 321, 500, 501, 502]
list1 = [8,9,7,71,72,73,102,103,130,200,201,267,311,312,432,433,434,512,513,514,515,556,783,784,785,786,787]
for viewId in (list1):
    filename = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz".format(objId, viewId)  #you have to be loged in on cuda4xl-01
    array = np.load(filename)['data']
    data.append(array[0])

# print len(data)

full_data = []
for viewId in range(view_number):
    filename = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_imagenet_mean_binaryproto/obj{}__{}.npz".format(objId, viewId)  #you have to be loged in on cuda4xl-01
    array = np.load(filename)['data']
    full_data.append(array[0])

# print(len(full_data))

dst = 0
for i in range(len(list1)):
    for j in range(len(list1)):
        temp = distance.euclidean(data[j], data[i])
        if dst < temp:
            dst = temp

print ("Max range of the Object {} in the group of same front/back views: {}".format(objId, dst))

dst_full = 0
for j in range(view_number):
    for i in range(view_number):
        temp1 = distance.euclidean(full_data[i], full_data[j])
        if dst_full < temp1:
            dst_full = temp1

print ("Max range of the Object {} in class: {} in".format(objId, dst_full))







