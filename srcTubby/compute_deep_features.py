###############################################################################
import ToolBOSCore.Util.Any as Any
import matplotlib.pyplot as plt
from mode_selection import mod
from procesing import process
from perform_classification import clas
import numpy as np
import caffe
###############################################################################

#CPU or GPU mode selection
mod('gpu', 3)

model_def = '/hri/storage/user/jradojev/Internship/Sebastian_net_data/testNet'   					 #model definition
model_weights = '/hri/storage/user/jradojev/Internship/Sebastian_net_data/pre-trained_end-to-end_iter_126600.caffemodel' #weights

#init the network
net= caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

transformer = process(net)

def doCompute(RTBOS):
 
    #get the image
    image_list = RTBOS.getInputRef(0) 			#RTBOS.getInputRef(0) is a list from the first input, 3 elements for 3 channels
    image_array = np.dstack(image_list)			#from a list to 3D array
    Any.log(0, "Image dim: " + str(image_array.shape))
  
    #compute the features 
    output = clas(transformer, net, image_array)	#classification 
    Any.log(0, "Predicted class: " + str(output['probs'][0].argmax()))
    
    #feature extraction 126 vector
    myFeatures = net.blobs['fc8_hri'].data		#taking the features out of the network
    #np.savetxt('/home/jradojev/PycharmProjects/Internship/rtbos_features.txt', myFeatures) #saves the features in .txt file
    Any.log(0,"Feature position: " + str(myFeatures.argmax()) + " Feature value: "+ str(myFeatures[0][myFeatures.argmax()]))
 
    #assign the features to the output
    features=RTBOS.getOutputRef(0)
    features[0][:]=myFeatures
    
    # fire the output
    RTBOS.fireOutputPort(0)
        
###############################################################################
