###############################################################################
import ToolBOSCore.Util.Any as Any
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
###############################################################################

# defining of k
n_neighbors=1
feature_memory = []  # this dimension can be taken from compute_deep_features.py ?
label_set = []  # incijalizacija niza


def doCompute(RTBOS):
    global feature_memory, label_set
    if not RTBOS.isInputNew(0):
        return
    # inputs
    feature_vector = RTBOS.getInputRef(0)[0]
    # print feature_vector
    category_labels = RTBOS.getInputRef(1)
    internal_status = RTBOS.getInputRef(2)

    print("Internal status: {}".format(internal_status))
    lenght=len(feature_memory)
    #print("Feature dimension: {}".format(feature_vector.shape))
    #print("Number of training examples: {}".format(lenght))
    print("Category labels: {}".format(category_labels))
    if (internal_status==0):
        if lenght != 0:

            neigh = NearestNeighbors(n_neighbors).fit(feature_memory)
            distances, indices = neigh.kneighbors(feature_vector)

            detected_categories = label_set[indices[0][0]]  # takes from the list of labels the label set which corresponds to the minimal euclidean distance;

            #print ("Position of the closest neighbour: {}".format(indices[0][0]))  # k neighbours can be taken instead of the argmax()

        else:
            detected_categories=np.zeros_like(category_labels)


        # assign the output
        categories = RTBOS.getOutputRef(0)
        categories[:] = detected_categories

            # fire the output
        RTBOS.fireOutputPort(0)


    # storing of feature vectors of objects
    else:
        feature_memory.append(np.copy(feature_vector[0]))  # memory with feature_vectors; izbaci one koji su blizu?
        label_set.append(np.copy(category_labels))  # memory with adequate label sets, confirmed from the user

###############################################################################
