###############################################################################
import ToolBOSCore.Util.Any as Any
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
###############################################################################

#defining of k 
k=1;
feature_memory=[]  #this dimension can be taken from compute_deep_features.py ? 
label_set=[]   #incijalizacija niza

def doCompute(RTBOS):

    global feature_memory, label_set
    if not RTBOS.isInputNew(0):
        return
    #inputs
    feature_vector = RTBOS.getInputRef(0)[0]
    #print feature_vector 
    category_labels = RTBOS.getInputRef(1)
    internal_status = RTBOS.getInputRef(2)

    lenght=len(feature_memory)
    print("Number of training examples: {}".format(lenght))
    if lenght != 0: 
        distances=np.empty([lenght])
        #print distances.dtype
        #computing the euclidean distance from all of the training examples
        for x in range(lenght): 
            neighbour_euclidean= distance.euclidean(feature_memory[x], feature_vector)
            distances[x]=neighbour_euclidean
            
        #print distances
        closest_neighbour_position=distances.argmin()
        detected_categories=label_set[closest_neighbour_position] #takes from the list of labels the label set which corresponds to the minimal euclidean distance;
        print ("Position of the closest neighbour: {}".format(closest_neighbour_position))                                   # k neighbours can be taken instead of the argmax()  
   
        #assign the output
        categories=RTBOS.getOutputRef(0)
        categories[:]=detected_categories
        
        # fire the output
        RTBOS.fireOutputPort(0)        
 
    #storing of feature vectors of objects 
    feature_memory.append(np.copy(feature_vector)) #memory with feature_vectors; izbaci one koji su blizu?
    label_set.append(np.copy(category_labels))          #memory with adequate label sets, confirmed from the user        
  

 
        
###############################################################################
