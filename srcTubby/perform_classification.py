import caffe

def clas(transformer,net, a):
    #this function performs clasification
    transformed_image = transformer.preprocess('data', a)
    net.blobs['data'].data[...] = transformed_image # copy the image data into the memory allocated for the net

    return net.forward()  # perform classification
