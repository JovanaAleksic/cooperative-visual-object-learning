from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import cv2
import matplotlib as mpl
import random
from sklearn.neighbors import NearestNeighbors
import pylab




view_number = 1200
sample_number = 100
objId = 6

#random samples
# view_list = random.sample(range(view_number), sample_number)
# view_list = range(view_number)

#saving list of random samples
# f = open('view_random_list100_object6.pckl', 'wb')
# pickle.dump(view_list, f)
# f.close()

#opening saved list
f = open('view_random_list100_object6.pckl', 'rb')
view_list = pickle.load(f)
f.close()


objId_list = [6, 7]

n_neighbors = 6   #this one affects the field size

average_color_list_b = []
average_color_list_g = []
average_color_list_r = []
for viewId in view_list:
    filename = '/hri/localdisk/stephanh/hri126plus/obj_converted/obj{}__{}.png'.format(objId, viewId)
    img = cv2.imread(filename)
    maskname = '/hri/localdisk/stephanh/hri126plus/mask/mask{}__{}.png'.format(objId, viewId)
    mask = cv2.imread(maskname)
    usepixels = np.where(mask > 0)
    average_color_list_b.append(img[usepixels[0], usepixels[1], 0].mean()) #average_color[0]
    average_color_list_g.append(img[usepixels[0], usepixels[1], 1].mean())
    average_color_list_r.append(img[usepixels[0], usepixels[1], 2].mean())



average_color_array_r = np.asarray(average_color_list_r)
average_color_array_g = np.asarray(average_color_list_g)

max_y = int(average_color_array_g.max())+1
min_y = int(average_color_array_g.min())
max_x = int(average_color_array_r.max())+1
min_x = int(average_color_array_r.min())

print(min_x, min_y, max_x, max_y )
feature_memory = np.vstack((average_color_array_r, average_color_list_g)).transpose()
neigh = NearestNeighbors(n_neighbors).fit(feature_memory)
distances, indices = neigh.kneighbors(feature_memory)

distances_mean = distances.transpose()[1:].transpose().mean()  #taking out zeros + mean of everything
# print(distances_mean) #distance.shape (100,5)

field_size = int(distances_mean)+1  #maybe it can be int(2*distance_mean), it doesn't have to be precise, better smaller number of neighbors
print(field_size)  #works now only with round numbers

y_range = (max_y-min_y) % field_size
x_range = (max_x-min_x) % field_size
y_condition = y_range == 0
x_condition = x_range == 0
if not (y_condition and x_condition):
    if not y_condition:
        add_y = field_size - y_range
        print(add_y)
        min_y -= add_y / 2
        max_y += add_y / 2
    if not x_condition:
        add_x = field_size - x_range
        print (add_x)
        min_x -= add_x / 2
        max_x += add_x / 2

print(min_x, min_y, max_x, max_y)

yd = int((max_y-min_y) / field_size)  #number of grid fields in y axis
xd = int((max_x-min_x) / field_size) #number of grid fields in x axis
grid =np.zeros((yd, xd))

print(yd, xd)

for viewId in view_list:
    # red training samples
    x_1 = int((average_color_list_r[view_list.index(viewId)]-min_x)/field_size)
    y_1 = int((average_color_list_g[view_list.index(viewId)]-min_y)/field_size)       #
    grid[y_1][x_1] += 1


grid /= (grid.max()/1.0)

side_x = np.linspace(min_x, max_x, (max_x-min_x)/field_size + 1) # +1 serves for plotting, has to be there
side_y = np.linspace(min_y, max_y, (max_y-min_y)/field_size + 1) #
X, Y = np.meshgrid(side_x, side_y)
fig = plt.figure()
fig.suptitle('Density field')
plt.pcolormesh(X, Y, grid, cmap='Reds')  # http://matplotlib.org/users/colormaps.html
pylab.ylim([min_y, max_y])
pylab.xlim([min_x, max_x])
plt.colorbar()

objId = 59
average_color_list_b1 = []
average_color_list_g1 = []
average_color_list_r1 = []
for viewId in view_list:
    filename = '/hri/localdisk/stephanh/hri126plus/obj_converted/obj{}__{}.png'.format(objId, viewId)
    img = cv2.imread(filename)
    maskname = '/hri/localdisk/stephanh/hri126plus/mask/mask{}__{}.png'.format(objId, viewId)
    mask = cv2.imread(maskname)
    usepixels = np.where(mask > 0)
    average_color_list_b1.append(img[usepixels[0], usepixels[1], 0].mean()) #average_color[0]
    average_color_list_g1.append(img[usepixels[0], usepixels[1], 1].mean())
    average_color_list_r1.append(img[usepixels[0], usepixels[1], 2].mean())



color = 'ro'
color1 = 'bo'
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
label_name = 'Object {}'.format(objId)
plt.plot(average_color_list_r1, average_color_list_g1, color1, label=label_name)
plt.plot(average_color_list_r, average_color_list_g, color, label=label_name)
fig.suptitle('RG: random 100 pictures ') #+labelname1
plt.xlabel('Red')
plt.ylabel('Green')
plt.legend()

count=0
for i in range(yd):
    for j in range(xd):
        if grid[i][j]!=0:
            count +=1

fill=count/(yd*xd)
print("Fill: {}%, count: {}".format(fill*100, count))

total_overlap_flag = 0
partial_overlap_flag1 = 0
partial_overlap_flag2 = 0

cover = 0
for viewId in view_list:
    x_1 = int((average_color_list_r1[view_list.index(viewId)]-min_x)/field_size)
    y_1 = int((average_color_list_g1[view_list.index(viewId)]-min_y)/field_size)       #
    if x_1 < xd and y_1 < yd:
        if grid[y_1][x_1] != 0:
            cover += 1
            grid[y_1][x_1] = 0
        else:
            partial_overlap_flag2 = 1 #partial_overlap_flag2 is active when not all of the samples in the new class are in the fields of the old one
    else:
        partial_overlap_flag2 = 1

overlap = cover / count
print("Overlap: {}%, cover: {}".format(overlap*100, cover))

if overlap == 1:
    total_overlap_flag = 1
else:
    partial_overlap_flag1 = 1  #partial_overlap_flag1 is active when not all of the fields in the existing class are covered with new class

print ("Total_overlap_flag: {}, partial_overlap_flag1: {}, partial_overlap_flag2: {}".format(total_overlap_flag, partial_overlap_flag1, partial_overlap_flag2))

plt.show()
