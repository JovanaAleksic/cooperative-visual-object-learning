###############################################################################
from __future__ import division
import ToolBOSCore.Util.Any as Any
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

###############################################################################

# defining of k
n_neighbors = 1
feature_memory = []
label_set = []
s = 1  # smoothening parameter
leave_out_treshold=0.1 #if distance to the neighbor is < than treshold feature_vector will be left out of feature_memory


def doCompute(RTBOS):
	global feature_memory, label_set, s
	if not RTBOS.isInputNew(0):
		return
	# inputs
	feature_vector = RTBOS.getInputRef(0)[0]
	# print feature_vector
	category_labels = RTBOS.getInputRef(1)
	internal_status = RTBOS.getInputRef(2)

	lenght = len(feature_memory)
	print("Number of training examples: {}".format(lenght))
	print('Internal status: {}'.format(internal_status))
	# print("Category labels dim: {}".format(category_labels.shape))

	if (internal_status==0):
		if (lenght > n_neighbors):
			P_1 = np.empty([category_labels.shape[0]])  # prior probability P(H_b^l), b=1 for yes, b=0 for no label
			P_0 = np.empty([category_labels.shape[0]])
			PE_1 = np.zeros((category_labels.shape[0], n_neighbors + 1))
			PE_0 = np.zeros((category_labels.shape[0], n_neighbors + 1))

			for x in range(category_labels.shape[0]):  # category labels 1-D array size 16
				a = 0
				for j in range(lenght):
					a = a + label_set[j][x]     #does not have to be computed ever time

				P_1[x] = (s + a) / (s * 2 + lenght)
				P_0[x] = 1 - P_1[x]


			neigh = NearestNeighbors(n_neighbors + 1).fit(feature_memory)
			distances, indices = neigh.kneighbors(feature_memory)

			for x in range(category_labels.shape[0]):
				c1 = np.empty(n_neighbors + 1)
				c2 = np.empty(n_neighbors + 1)
				for z in range(lenght):
					beta = 0
					for n in range(n_neighbors):
						if label_set[indices[z][n + 1]][x] == 1:
							beta = beta + 1
					if label_set[z][x] == 1:
						c1[beta] = c1[beta] + 1
					else:
						c2[beta] = c2[beta] + 1
				for j in range(n_neighbors + 1):
					sc1 = 0
					sc2 = 0
					for p in range(n_neighbors + 1):
						sc1 = sc1 + c1[p]
						sc2 = sc2 + c2[p]
					PE_1[x][j] = (s + c1[j]) / (s * (n_neighbors + 1) + sc1)
					PE_0[x][j] = (s + c2[j]) / (s * (n_neighbors + 1) + sc2)


			distances_t, indices_t = neigh.kneighbors(feature_vector)
			for x in range(category_labels.shape[0]):
				betax = 0
				for y in range(n_neighbors):
					if label_set[indices_t[0][y]][x] == 1:
						betax = betax + 1

			detected_categories = np.empty(category_labels.shape[0])
					# print ("Dimensions of detected categories: {}".format(detected_categories.shape))

			for x in range(category_labels.shape[0]):
				if (P_1[x] * PE_1[x][betax] > P_0[x] * PE_0[x][betax]):
					detected_categories[x] = 1
				else:
					detected_categories[x] = 0

		else:
			detected_categories=np.zeros_like(category_labels)

		# assign the output
		categories = RTBOS.getOutputRef(0)
		categories[:] = detected_categories
		print("Detected categories: {}".format(detected_categories))

				# fire the output
		RTBOS.fireOutputPort(0)

		#if distances_t[0][0] > leave_out_treshold:	# storing of feature vectors of objects
			#feature_memory.append(np.copy(feature_vector[0]))  # memory with feature_vectors
			#label_set.append(np.copy(category_labels))  # memory with adequate label sets, CONFIRMED from the user!
	else:
		feature_memory.append(np.copy(feature_vector[0]))  # memory with feature_vectors
		label_set.append(np.copy(category_labels))
	###############################################################################
