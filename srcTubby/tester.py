###############################################################################
from __future__ import division
import ToolBOSCore.Util.Any as Any
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

###############################################################################


n_neighbors=1

t=[[0,0,0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2,2,2],
[3,3,3,3,3,3,3,3,3,3], [4,4,4,4,4,4,4,4,4,4], [5,5,5,5,5,5,5,5,5,5],
[6,6,6,6,6,6,6,6,6,6], [7,7,7,7,7,7,7,7,7,7], [8,8,8,8,8,8,8,8,8,8], [9,9,9,9,9,9,9,9,9,9]]

l=[[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[1,0,1,0,0,0,0,0,0,0],
   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],
   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

test_example=np.array([[0,0,0,0.1,0,0,0,0,0,0]])
s=1 #smoothening parameter

lenght = len(t)
#print("Number of training examples: {}".format(lenght))


if lenght != 0:
   P_1 = np.empty(10)  # prior probability P(H_b^l), b=1 for yes, b=0 for no label
   P_0 = np.empty(10)
   PE_1 = np.zeros((10,n_neighbors+1))
   PE_0 = np.zeros((10,n_neighbors+1))
   for x in range(10):
      a = 0
      for j in range(lenght):
         a = a + l[j][x]

      #print("a: {}".format(a))
      P_1[x] = (s + a) / (s * 2 + lenght)
      P_0[x] = 1 - P_1[x]
   #print("Probabilities: {}".format(P_1))

   neigh = NearestNeighbors(n_neighbors+1).fit(t) #mistake! calculates itself in the neighbors, wrong!
   distances, indices = neigh.kneighbors(t)
   #print ("New")
   #print indices, distances

   for x in range(10):
      c1 = np.empty(n_neighbors+1)
      c2 = np.empty(n_neighbors+1)
      for z in range(lenght):
         beta = 0
         for n in range(n_neighbors):  # n_neighbors or n_neighbors+1 ??? -1
            if l[indices[z][n+1]][x] == 1:  # check index values!
               beta = beta + 1
         if l[z][x] == 1:
            c1[beta] = c1[beta] + 1
         else:
            c2[beta] = c2[beta] + 1

      print c1
      for j in range(n_neighbors+1):
         sc1 = 0
         sc2 = 0
         for p in range(n_neighbors+1):  # -1
            sc1 = sc1 + c1[p]
            sc2 = sc2 + c2[p]
         PE_1[x][j] = (s + c1[j]) / (s * (n_neighbors+1) + sc1)  # +1
         PE_0[x][j] = (s + c2[j]) / (s * (n_neighbors+1) + sc2)  # +1


   #print PE_1
   #print PE_0
   distances_t, indices_t = neigh.kneighbors(test_example)
   #print indices_t

   detected_categories = np.empty(10)
   for x in range(10):
      betax = 0
      for y in range(n_neighbors):
         if l[indices_t[0][y]][x] == 1:
            betax = betax + 1
      print betax
      if ((P_1[x] * PE_1[x][betax]) > (P_0[x] * PE_0[x][betax])):
         detected_categories[x] = 1
      else:
         detected_categories[x] = 0

   # assign the output
   print("Detected categories: {}".format(detected_categories))

