import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

# data = np.load('obj6__0.npz')
# array = data['data']
# # print(array)

view_number = 1200

maximum_indicators = np.empty(view_number, dtype=np.int)
objId = 97
for viewId in range(view_number):
    filename = "/hri/localdisk/stephanh/hri126plus/caffenet_fc8_ilsvrc_2012_mean_center/obj{}__{}.npz".format(objId, viewId)  #you have to be loged in on cuda4xl-01
    data = np.load(filename)['data']
    maximum_indicators[viewId] = data[0].argmax()

filename = "/hri/localdisk/stephanh/hri126plus/alexnet_fc8/obj6__{}.npz"
#
# fig = plt.figure(1)
# plt.plot(range(len(array[0])), array[0], 'k')
# fig.suptitle('Object 6')

# fig = plt.figure(1)
# plt.plot(range(view_number), maximum_indicators, 'ko')
# fig.suptitle('Max indicators')

counts = np.bincount(maximum_indicators)
counts_sorted = np.argsort(counts)
# indicator_max=np.argmax(counts)
# print counts

class_names = []
with open('/hri/storage/user/jradojev/caffe/data/ilsvrc12/synset_words.txt') as class_file:
    for line in class_file:
        class_names.append(line[10:-1])

for i in range(10):
    print("{}) Indicator: {}, Counts: {}, Name: {}".format(i, counts_sorted[-i-1], counts[counts_sorted[-i-1]], class_names[counts_sorted[-i-1]]))




# data = np.load('/hri/localdisk/stephanh/hri126plus/alexnet_fc8/obj93__0.npz')
# array = data['data']
#
# fig = plt.figure(2)
# plt.plot(range(len(array[0])), array[0], 'b')
# fig.suptitle('Object')

plt.show()
