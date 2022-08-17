from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import cv2
import matplotlib as mpl
import random
from sklearn.neighbors import NearestNeighbors
import pylab
import math
from overlap_measures import overlap_measures
from overlap_detection import overlap_detection
from actions import confidence_computation
from get_views import get_views
from PIL import Image
########################################################################################################################
# DEFINITION OF COLORS

def prRed(prt): print("\033[31m{}\033[00m" .format(prt))
def prGreen(prt): print("\033[32m{}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m{}\033[00m" .format(prt))
def prBlue(prt): print("\033[34m{}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m{}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m{}\033[00m" .format(prt))
def prLightGray(prt): print("\033[37m{}\033[00m" .format(prt))
def prBlack(prt): print("\033[30m{}\033[00m" .format(prt))


black = '\033[30m'
red = '\033[31m'
green = '\033[32m'
orange = '\033[33m'
blue = '\033[34m'
purple = '\033[95m'
darkcyan = '\033[36m'
lightgrey = '\033[37m'
darkgrey = '\033[90m'
lightred = '\033[91m'
lightgreen = '\033[92m'
yellow = '\033[93m'
lightblue = '\033[94m'
pink = '\033[95m'
cyan = '\033[96m'
end = "\033[00m"
########################################################################################################################
# PARAMETERS

view_number = 1200
view_list = range(view_number)
batch = 100                     #number of samples coming together before processing
objId_list = [6, 7, 70, 55, 80]    #list of objects in the demonstration, possible to stop at any point
neuron_id = [200, 722]          #which neurons output we take for 2D case
n_neighbors = 5                 #k number in paper, how many neighbors is consider for confidence measure
treshold = 0.01                 #size of kappa, gamma, delta to be considered to be in the overlappigng region
treshold1 = 0.7                 #confidence high enough to say it is something
treshold_main = 0.8             #at what confidence level I can say it is something
flag_overlap = 0                #flag which is activated when a test object is not new group, either stated by user or the system
treshold_low = 0.3                #in case we have only  object in memory, if overlap is big enough it is old object

treshold1 = 8  #maximum 10
treshold2 = 30

########################################################################################################################
#CUSTOM FUNCTIONS

def length(s):                  #number of different objects previuously seen
    return len(list(set(s)))



def do_new(x, y, objects, label, h, entry, flag_overlap):
    print(purple + "I think this can be a new object. These are all objects which I have seen so far!" + end)
    for i in list(set(objects)):  # list of different learned objects
        index = objects.index(i)  # first index in object list for every object name
        obj = objId_list[index]  # objId for that index
        image_name = "/hri/localdisk/stephanh/hri126plus/obj/obj{}__{}.png".format(obj, 0)
        Image.open(image_name).show()  # zero-th view

    answer = input(purple + "Am I right? Answer with 0 for no, and 1 for yes!" + end)
    if answer == 1:
        answer = input(purple + "Can you tell me what it is?" + end)
        objects.append(answer)
        label.extend([answer] * batch)
        answer = input(
            purple + "Since this is a new object for me would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
        while answer == 1 and (h < view_number):
            print(lightgrey + "I am getting more views of Object {}".format(objId) + end)
            x, y = get_views(objId, range(h, batch + h), x, y, neuron_id)
            h += batch
            label.extend([objects[-1]] * batch)
            answer = input(
                purple + "Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
        entry = 0
    else:
        if length(objects) == 1:
            print(
            purple + "Then it is {}. This means that I haven't seen enough views of {} in previous sessions. Please make sure to show me all views now." + end).format(
                objects[0], objects[0])
            objects.append(objects[0])
            label.extend([objects[0]] * batch)

            answer = input(purple + "Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
            while answer == 1 and (h < view_number):
                print(lightgrey + "I am getting more views of Object {}".format(objId) + end)
                x, y = get_views(objId, range(h, batch + h), x, y, neuron_id)
                h += batch
                label.extend([objects[0]] * batch)
                answer = input(
                    purple + "Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
            entry = 0
        else:
            flag_overlap = 1
    return x, y, objects, label, h, entry, flag_overlap   #probably you don't need x and y here



def do_old_first(x,y, objects, label, h, entry, flag_overlap):
    print(purple + "I think this is {}." + end).format(objects[0])
    answer = input(purple + "Is this correct? Answer with 0 for no, and 1 for yes!" + end)
    if answer == 1:
        objects.append(objects[0])
        label.extend([objects[0]] * h)
        entry = 0
    else:
        answer = input(purple + "Can you tell me what it is?" + end)
        objects.append(answer)
        label.extend([answer] * batch)
        print(purple + "This two objects look same to me!" + end)
        index = objects.index(objects[0])  # first index in object list for every object name
        obj = objId_list[index]
        image_name = "/hri/localdisk/stephanh/hri126plus/obj/obj{}__{}.png".format(obj, 0)
        Image.open(image_name).show()  # zero-th view
        answer = input(purple + "Are they similar to you? Answer with 0 for no, and 1 for yes!" + end)
        if answer == 1:
            prPurple(
                "Then please show me the side of that object which is the most distinctive one with respect to the other object.")
            answer = input(
                purple + "Would you like to show me that side? Answer with 0 for no, and 1 for yes!" + end)
            while answer == 1 and (h < view_number):
                print(lightgrey + "I am getting more views of Object {}".format(objId) + end)
                x, y = get_views(objId, range(h, batch + h), x, y, neuron_id)
                h += batch
                label.extend([objects[0]] * batch)
                answer = input(
                    purple + "Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
            entry = 0
        else:
            prPurple("Then please show me as much views as possible.")
            answer = input(
                purple + "Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
            while answer == 1 and (h < view_number):
                print(lightgrey + "I am getting more views of Object {}".format(objId) + end)
                x, y = get_views(objId, range(h, batch + h), x, y, neuron_id)
                h += batch
                label.extend([objects[0]] * batch)
                answer = input(
                    purple + "Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
            entry = 0

    return x, y, objects, label, h, entry, flag_overlap

########################################################################################################################
#PLOT

plt.axis([1, 10, 0, 1])
plt.ion()


########################################################################################################################
# MAIN

label = []
objects = []
x = []
y = []
h =batch

for objId in objId_list:
    entry = input(purple + "Can you show me a new object? Answer with 0 for no, and 1 for yes!" + end)
    plt.close("all")
    y1 = [0, 0]
    ig = 1
    if entry == 1:
        print(lightgrey + "I am getting views of Object {}".format(objId) + end)
        label_name = 'Object {}'.format(objId)   #Probably i don't need it

        if len(label) == 0:
            x, y = get_views(objId, view_list, x, y, neuron_id)
            x_array = np.asarray(x)
            y_array = np.asarray(y)
            answer = input(purple + "This is my first object so I don't know what this is? Please tell me?" + end)
            objects.append(answer)                       #adds learned objects to the list
            label.extend([answer]*len(view_list))        #assigns a list of names

            answer = input(
                purple + "Since this is a new object for me would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
            while (answer == 1) and (h < view_number):
                print(lightgrey + "I am getting more views of Object {}".format(objId) + end)
                x, y = get_views(objId, range(h, batch + h), x, y, neuron_id)
                h += batch
                label.extend([objects[-1]] * batch)
                prGreen("Add self-monitoring part")
                answer = input(
                    purple + "Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)

            feature_memory = np.vstack((x_array, y_array)).transpose()
            neigh = NearestNeighbors(n_neighbors).fit(feature_memory)
        elif length(objects) == 1:
            x = []
            y = []
            x, y = get_views(objId, range(batch), x, y, neuron_id)
            h = batch
            x_test_array = np.asarray(x)
            y_test_array = np.asarray(y)
            test = np.vstack((x_test_array, y_test_array)).transpose()

            kappa1, gamma1, delta1, indices = overlap_measures(neigh, n_neighbors, feature_memory, test)
            outlier, sparse_region, dense_overlap, index_total = \
                overlap_detection(treshold1, treshold2, kappa1, gamma1, delta1)
            confidence, new, measure = confidence_computation(outlier, sparse_region, dense_overlap)
            fig=plt.figure()
            fig.canvas.manager.window.attributes('-topmost', 1)
            y1 = [y1[1], confidence]
            i1 = [ig - 1, ig]
            plt.plot(i1, y1, '-bo')
            plt.pause(0.05)
            ig+=1

            prCyan("Confidence:{}".format(round(confidence, 2)))
            prCyan("Measure:{}".format(round(measure, 2)))



            if confidence > treshold_main and new == 1:
                x, y, objects, label, h, entry, flag_overlap = do_new(x, y, objects, label, h, entry, flag_overlap)

            elif confidence > treshold_main and new == 0:
                x, y, objects, label, h, entry, flag_overlap = do_old_first(x, y, objects, label, h, entry,
                                                                            flag_overlap)

            else:
                answer = input(
                    purple + "I am confused. Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
                while answer == 1 and (h < view_number):
                    print(lightgrey + "I am getting more views of Object {}".format(objId) + end)
                    x, y = get_views(objId, range(h, batch + h), x, y, neuron_id)
                    h += batch
                    x_test_array = np.asarray(x)
                    y_test_array = np.asarray(y)
                    test = np.vstack((x_test_array, y_test_array)).transpose()

                    kappa1, gamma1, delta1, indices = overlap_measures(neigh, n_neighbors, feature_memory, test)
                    outlier, sparse_region, dense_overlap, index_total = \
                        overlap_detection(treshold1, treshold2, kappa1, gamma1, delta1)
                    confidence, new, measure = confidence_computation(outlier, sparse_region, dense_overlap)
                    y1 = [y1[1], confidence]
                    i1 = [ig - 1, ig]
                    plt.plot(i1, y1, '-bo')
                    plt.pause(0.05)
                    ig += 1
                    if confidence > treshold_main:
                        prCyan("Confidence:{}".format(round(confidence, 2)))
                        prCyan("Measure:{}".format(round(measure, 2)))
                        break

                    prCyan("Confidence:{}".format(round(confidence, 2)))
                    prCyan("Measure:{}".format(round(measure, 2)))

                    answer = input(
                        purple + "I still cannot decide. Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)

                if new == 1:
                    x, y, objects, label, h, entry, flag_overlap = do_new(x, y, objects, label, h, entry,
                                                                          flag_overlap)
                else:
                    x, y, objects, label, h, entry, flag_overlap = do_old_first(x, y, objects, label, h, entry,
                                                                                flag_overlap)

        else:
            x = []
            y = []
            x, y = get_views(objId, range(batch), x, y, neuron_id)
            h=batch
            x_test_array = np.asarray(x)
            y_test_array = np.asarray(y)
            test = np.vstack((x_test_array, y_test_array)).transpose()

            kappa1, gamma1, delta1, indices = overlap_measures(neigh, n_neighbors, feature_memory, test)
            outlier, sparse_region, dense_overlap, index_total =\
                overlap_detection(treshold1, treshold2, kappa1, gamma1, delta1)
            confidence, new, measure = confidence_computation(outlier, sparse_region, dense_overlap)

            if confidence > treshold_main and new == 1:
                fig = plt.figure()
                fig.canvas.manager.window.attributes('-topmost', 1)
                y1 = [y1[1], confidence]
                i1 = [ig - 1, ig]
                plt.plot(i1, y1, '-bo')
                plt.pause(0.05)
                ig += 1
                prCyan("Confidence:{}".format(round(confidence, 2)))
                prCyan("Measure:{}".format(round(measure, 2)))
                x, y, objects, label, h, entry, flag_overlap = do_new(x, y, objects,label,h,entry,flag_overlap)

            else:
                flag_overlap = 1

            if flag_overlap == 1:
                #getting the number of overlapping classes
                s = []
                print (lightgrey + "Total number of test samples h: {}, index_total: {}".format(h, len(index_total)) + end)
                for i in range(len(index_total)):
                    if index_total[i] == 1:
                        for j in range(n_neighbors):
                            ix = indices[i][j]
                            s.append(label[ix])   #
                cles_overlap = list(set(s))
                nb = length(s)      #we get number of overlaping classes here
                pro=np.zeros((nb))
                for i in range(int(len(s)/n_neighbors)):
                    c=length(s[(i*n_neighbors):((i+1)*n_neighbors)])
                    if c==1:
                        v = cles_overlap.index(s[i*n_neighbors])
                        pro[v] += 1
                    elif c == pro.shape[0]:
                        pro += 1
                    else:         #This is the case where you have for certain sample 2 neighboring classes but overall there is more overlap classe
                        p=list(set(s[(i*n_neighbors):((i+1)*n_neighbors)]))
                        for j in p:
                            v = cles_overlap.index(j)
                            pro[v] += 1

                print (lightgrey + "Max pro: {}".format(pro.max()/(outlier+sparse_region+dense_overlap)) + end)
                confidence = ((max(pro.max(), outlier))/(outlier+sparse_region+dense_overlap)-1/(nb+1)) * ((nb+1)/nb)
                fig = plt.figure()
                fig.canvas.manager.window.attributes('-topmost', 1)
                y1 = [y1[1], confidence]
                i1 = [ig - 1, ig]
                plt.plot(i1, y1, '-bo')
                plt.pause(0.05)
                ig += 1
                prCyan("Confidence: {}".format(round(confidence, 2)))
                print(lightgrey + "Number of overlapping classes: {}".format(nb) + end)
                # objects.append(objects[0])
                # label.extend([objects[0]] * min_sample)
                # baseline = round(1/nb, 2)
                if confidence < treshold_main:
                    answer = input(
                        purple + "I am confused. Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)
                    while answer == 1 and (h < view_number):
                        print(lightgrey + "I am getting more views of Object {}".format(objId) + end)
                        x, y = get_views(objId, range(h, batch + h), x, y, neuron_id)
                        h += batch
                        x_test_array = np.asarray(x)
                        y_test_array = np.asarray(y)
                        test = np.vstack((x_test_array, y_test_array)).transpose()

                        kappa1, gamma1, delta1, indices = overlap_measures(neigh, n_neighbors, feature_memory, test)
                        outlier, sparse_region, dense_overlap, index_total = \
                            overlap_detection(treshold1, treshold2, kappa1, gamma1, delta1)
                        s = []
                        print (
                        lightgrey + "Total number of samples h: {}, index_total: {}".format(h, len(index_total)) + end)
                        for i in range(len(index_total)):
                            if index_total[i] == 1:
                                for j in range(n_neighbors):
                                    ix = indices[i][j]
                                    s.append(label[ix])  # PROBLEM HERE, list index out of range
                        cles_overlap = list(set(s))
                        nb = length(s)
                        pro = np.zeros((nb))
                        for i in range(int(len(s) / n_neighbors)):
                            c = length(s[(i * n_neighbors):((i + 1) * n_neighbors)])
                            if c == 1:
                                v = cles_overlap.index(s[i * n_neighbors])
                                pro[v] += 1
                            elif c == pro.shape[0]:
                                pro += 1
                            else:  # This is the case where you have for a test sample 2 neighboring classes but overall there is more overlap classe
                                p = list(set(s[(i * n_neighbors):((i + 1) * n_neighbors)]))
                                for j in p:
                                    v = cles_overlap.index(j)
                                    pro[v] += 1
                        confidence = ((max(pro.max(), outlier)) / (outlier + sparse_region + dense_overlap) - 1 / (
                        nb + 1)) * ((nb + 1) / nb)
                        y1 = [y1[1], confidence]
                        i1 = [ig - 1, ig]
                        plt.plot(i1, y1, '-bo')
                        plt.pause(0.05)
                        ig += 1
                        if confidence > treshold_main:
                            prCyan("Confidence:{}".format(round(confidence, 2)))
                            prCyan("Measure:{}".format(round(measure, 2)))
                            break

                        prCyan("Confidence:{}".format(round(confidence, 2)))
                        prCyan("Measure:{}".format(round(measure, 2)))

                        answer = input(
                            purple + "I still cannot decide. Would you like to show me more views? Answer with 0 for no, and 1 for yes!" + end)

                    if outlier > pro.max():
                        flag_overlap = 0
                        x, y, objects, label, h, entry, flag_overlap = do_new(x, y, objects, label, h, entry,
                                                                              flag_overlap)
                        if flag_overlap == 1:
                            prPurple("Then it is {}".format(cles_overlap[pro.argmax()]))
                            objects.append(cles_overlap[pro.argmax()])
                            label.extend([objects[-1]] * h)
                            entry = 0

                    else:
                        prPurple("It is:{}".format(cles_overlap[pro.argmax()]))
                        objects.append(cles_overlap[pro.argmax()])
                        label.extend([objects[-1]] * h)
                        entry = 0
                else:
                    if outlier > pro.max():
                        flag_overlap = 0
                        x, y, objects, label, h, entry, flag_overlap = do_new(x, y, objects, label, h, entry,
                                                                              flag_overlap)
                        if flag_overlap == 1:
                            prPurple("Then it is {}".format(cles_overlap[pro.argmax()]))
                            objects.append(cles_overlap[pro.argmax()])
                            label.extend([objects[-1]] * h)
                            entry = 0

                    else:
                        prPurple("It is:{}".format(cles_overlap[pro.argmax()]))
                        objects.append(cles_overlap[pro.argmax()])
                        label.extend([objects[-1]] * h)
                        entry = 0




            feature_memory = np.hstack((feature_memory.transpose(), np.vstack((x_test_array, y_test_array)))).transpose()
            neigh = NearestNeighbors(n_neighbors).fit(feature_memory)

    else:
        entry = 0
        break
if entry == 0:
        print(blue + "Thank you for helping me. End of session!" + end)


print(green + "I have learned objects: {}".format(list(set(objects)))) #removes duplicates


