
import numpy as np
import tensorflow as tf
import math as math
#from matplotlib import pyplot as plt






def down_conv(inTens,outDepth,iLayer):

	kernelSize = (5,5)
	kernelDepth = inTens.get_shape()[3]

	W = tf.get_variable('W_downConv' + str(iLayer),(kernelSize[0],kernelSize[1],kernelDepth,outDepth),initializer=tf.random_normal_initializer(0,0.02))			
	b = tf.get_variable('b_downConv' + str(iLayer),outDepth,initializer=tf.constant_initializer(0.0))
	
	h = tf.nn.conv2d(inTens,W,strides=(1,2,2,1),padding='SAME',name='downConv_' + str(iLayer)) + b

	outTens = tf.nn.relu(h,name='out_downConv_' + str(iLayer))

	return outTens

def up_conv(inTens,outDepth,iLayer):

	kernelSize = (5,5)
	kernelDepth = inTens.get_shape()[3]	

	W = tf.get_variable('W_upConv' + str(iLayer),(kernelSize[0],kernelSize[1],outDepth,kernelDepth),initializer=tf.random_normal_initializer(0,0.02))			
	b = tf.get_variable('b_upConv' + str(iLayer),outDepth,initializer=tf.constant_initializer(0.0))
	
	#conv transpose doesn't allow unspecified dimensions so make sure these are calculated at runtime to allow variable batch and image size
	#outShape = tf.pack([tf.shape(inTens)[0],tf.shape(inTens)[1] * 2,tf.shape(inTens)[2] * 2,outDepth])


	h = tf.nn.conv2d_transpose(inTens,W,(tf.shape(inTens)[0],tf.shape(inTens)[1] * 2,tf.shape(inTens)[2] * 2,outDepth),strides=(1,2,2,1),padding='SAME',data_format='NHWC',name='upConv' + str(iLayer))  + b

	#outTens = tf.nn.relu(h,name='out_upConv_' + str(iLayer))
	#outTens = tf.scalar_mul(maxDisp,tf.nn.tanh(h),name='out_upConv_' + str(iLayer)) #We need to allow for negative displacements so we use tanh
	#outTens = tf.multiply(maxDisp,tf.nn.tanh(h),name='out_upConv_' + str(iLayer)) #We need to allow for negative displacements so we use tanh
	outTens = tf.nn.tanh(h,name='out_upConv_' + str(iLayer)) #We need to allow for negative displacements so we use tanh

	return outTens	

def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
    rep = tf.cast(rep, 'int32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

def interpolate(im, x, y, out_size,opName):

		#Credit due to daviddao - taken from tensorflow STN implementation

        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')        
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        #x = (x + 1.0)*(width_f) / 2.0
        #y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = _repeat(tf.range(num_batch)*dim1, tf.reduce_prod(out_size))
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.pack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        #output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id],name=opName)
        output_flat = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        output = tf.reshape(output_flat,tf.shape(im),name=opName)
        return output


def get_image_pair_batch(trainIms,testIms,baseBatchSize):

	#Returns batches of sequential image pairs with an equal number from each image series.
	

	nTrainTS = len(trainIms)
	nTestTS = len(testIms)

	currTrainBatch = [ [] for _ in range(nTrainTS)]
	for iTS in range(0,nTrainTS):
		i_t1 = np.random.randint(0,trainIms[iTS].shape[2]-2,baseBatchSize)		
		currBatch_t1 = trainIms[iTS][:,:,i_t1,:]
		currBatch_t2 = trainIms[iTS][:,:,i_t1+1,:]

		currTrainBatch[iTS] = np.concatenate([currBatch_t1,currBatch_t2],3)

	#Combine into a single batch
	trainImBatch = np.concatenate(currTrainBatch,2)
	trainImBatch = np.transpose(trainImBatch,(2,0,1,3))

	
	#Get pairs of sequental frames from each series	
	currTestBatch = [ [] for _ in range(nTestTS)]
	for iTS in range(0,nTestTS):
		i_t1 = np.random.randint(0,testIms[iTS].shape[2]-2,baseBatchSize)		
		currBatch_t1 = testIms[iTS][:,:,i_t1,:]
		currBatch_t2 = testIms[iTS][:,:,i_t1+1,:]

		currTestBatch[iTS] = np.concatenate([currBatch_t1,currBatch_t2],3)

	#Combine into a single batch
	testImBatch = np.concatenate(currTestBatch,2)
	testImBatch = np.transpose(testImBatch,(2,0,1,3))

	return trainImBatch,testImBatch

def show_results(ims,dispMaps,iIm):

	cf = plt.figure()
	cf.add_subplot(2,2,1)
	plt.imshow(np.concatenate([ims[iIm,:,:,:],np.zeros(ims.shape[1:3]  + (1,))],2 ))
	cf.add_subplot(2,2,2)
	dispMapShow = dispMaps[iIm,:,:,:].copy()
	dispMapShow = dispMapShow - np.amin(dispMapShow)
	dispMapShow = dispMapShow / np.max(dispMapShow)
	plt.imshow(np.concatenate([np.zeros(dispMaps.shape[1:3]  + (1,)),dispMapShow] ,2 ) )
	cf.add_subplot(2,2,3)
	plt.imshow(np.concatenate([ims[iIm+1,:,:,:],np.zeros(ims.shape[1:3]  + (1,))],2 ))	
	cf.add_subplot(2,2,4)
	dispMapShow = dispMaps[iIm+1,:,:,:].copy()
	dispMapShow = dispMapShow - np.amin(dispMapShow)
	dispMapShow = dispMapShow / np.max(dispMapShow)
	plt.imshow(np.concatenate([np.zeros(dispMaps.shape[1:3]  + (1,)),dispMapShow] ,2 ) )

	plt.show()
