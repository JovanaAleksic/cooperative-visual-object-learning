import caffe

def mod(compute_mode, device_id):
    #with this function we are setting cpu or gpu mode, compute_mode is choice between 'cpu' or 'gpu',
    #and device_id presents device number in case of gpu

    if  compute_mode=='gpu':
        caffe.set_mode_gpu()
        caffe.set_device(device_id)   # set to GPU

    elif compute_mode=='cpu':
	 caffe.set_mode_cpu() #set to CPU

    else: 
	print 'Unknown compute_mode:', compute_mode
