import caffe

def process(net):
    #this function is transforming all images which are going to the network,
    #and provides an adequate input which can be processed in the net

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # Height * Width * Channels is what image library provides,
    #but caffe expects C * H * W, so this command is only replacing the order of dimensions

    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # set the size of the input( we can skip this if we're happy
    #  with the default, we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(1,  # batch size
                              3,  # 3-channel (BGR) images
                              227, 227)

    return transformer
