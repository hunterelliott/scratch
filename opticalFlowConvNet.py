import numpy as np
import tensorflow as tf
import os
import tf_readImages as tfr
#from TPS_STN import TPS_STN
import ofnOps as ofn
from matplotlib import pyplot as plt
import time


# --------- parameters ---------

# -- Model parameters
layerWidth = (16, 32, 64) #Num feature maps per conv layer (order reversed for up-conv layers)
nLayers = len(layerWidth)
maxDisp = 10

# -- Optimization parameters
nIters = int(1e5)
baseLR = 1e-4#Base learning rate
baseBatchSize = 6 #Number of image pairs per image series to include in a batch

# -- logging parameters
logInterval = 10 #Log loss etc every n iterations
saveInterval = int(5e2) #Save model evern n iterations


# ---------- Input ----------

trainParentDir = '/home/hunter/Desktop/TEMP_LOCAL/Data/Arp23/Train'
testParentDir = '/home/hunter/Desktop/TEMP_LOCAL/Data/Arp23/Test'

#outDir = '/home/hunter/Desktop/TEMP_LOCAL/Data/Arp23/Training_Experiments'
outDir = '/home/hunter/Desktop/TEMP_LOCAL/Data/Arp23/Training_Experiments/TestRun_256x256_Batch6_Run1'
baseOutName = 'Arp_256x256' #Base file name for checkpoints (iteration number will be appended)

trainIms,testIms = tfr.readAllTimeSeriesImages(trainParentDir,testParentDir)
nTrainTS = len(trainIms)
nTestTS = len(testIms)
batchSizeTrain = nTrainTS * baseBatchSize
batchSizeTest = nTestTS * baseBatchSize

#trainIms = np.concatenate(trainIms,2)
nImChan = trainIms[0].shape[3]
#nTrainTot = trainIms.shape[2]



# ---------- Model creation ----------



data = tf.placeholder("float",shape=[None,None,None,nImChan*2],name='data') #Allow variable image size. Will this work with different batch sizes??
#data = tf.placeholder("float",shape=[batchSizeTrain,trainIms.shape[0],trainIms.shape[1],nImChan*2],name='data') #Will this work with resizing batch and image later????
#nImChan = tf.shape(data)[3] 


# -- down-convolution layers

print('Creating encoding downconvolution layers')
currInput = data
for iLayer in range(0,nLayers):

	currInput = ofn.down_conv(currInput,layerWidth[iLayer],iLayer)

# -- up-convolution layers

print('Creating decoding upconvolution layers')
#Use the reversed dimensionality progression with the last layer 2D as we will use it as a displacement vector field
outLayerWidth = layerWidth[-2::-1] + (2,)
nOutLayers = len(outLayerWidth)

for iLayer in range(0,nOutLayers):

	currInput = ofn.up_conv(currInput,outLayerWidth[iLayer],iLayer)


#forward = tf.identity(currInput,name='forward')
forward = tf.multiply(currInput,maxDisp,name='forward') #Expand maximum displacement from tanh range


# ----- Loss definition ----------
#Loss is L2 norm of input image displaced by vector field output of network

#TEMP - move all of this to ops module after it's been fully validated
xI = 0.0
yI = 0.0
xE = tf.to_float(tf.shape(forward)[1]) #OR shape -1?? interp source has 0-width....
yE = tf.to_float(tf.shape(forward)[2])
nX = tf.shape(forward)[1]
nY = tf.shape(forward)[2]

X,Y = tf.meshgrid(tf.linspace(xI,xE,nX),tf.linspace(yI,yE,nY))
XY = tf.pack([X,Y],2)

#Split timepoints - one is interpolated to attempt to predict the other
data_t1 = tf.slice(data,[0, 0, 0, 0],tf.pack([tf.shape(data)[0],tf.shape(data)[1],tf.shape(data)[2],nImChan]))
data_t2 = tf.slice(data,[0, 0, 0, 1],tf.pack([tf.shape(data)[0],tf.shape(data)[1],tf.shape(data)[2],nImChan]))

#Interpolate at positions displaced by the network output
UV = tf.add(XY,forward)
#UV = tf.add(XY,tf.zeros(tf.shape(forward))) #FOR TESTING
u = tf.reshape(UV[:,:,:,0], [-1])
v = tf.reshape(UV[:,:,:,1], [-1])

data_t1_prime = ofn.interpolate(data_t1, u, v, tf.shape(data)[1:3],'data_t1_prime')

#Loss is L2 norm of difference between propagated t1 image and t2 image
loss_train = tf.nn.l2_loss(data_t2 - data_t1_prime,'L2_loss_train')
#loss_test = tf.nn.l2_loss(data_t2 - data_t1_prime,'L2_loss_train')

#Use summary to log loss over time
tf.summary.scalar("loss",loss_train)

summary_op = tf.summary.merge_all()

## ------- optimization ----------

print("Starting optimization...")
startTime = time.time();	


optimizer = tf.train.AdamOptimizer(baseLR).minimize(loss_train)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


summary_writer = tf.summary.FileWriter(outDir,graph=tf.get_default_graph())
with tf.device('cpu:0'):
	saver = tf.train.Saver()


for iIter in range(0,nIters):


	trainImBatch,testImBatch = ofn.get_image_pair_batch(trainIms,testIms,baseBatchSize)

	optimizer.run(feed_dict={data:trainImBatch})
	
	if iIter % logInterval == 0:
		[lossPerIm,summary] = sess.run([loss_train,summary_op],feed_dict={data:trainImBatch})
		summary_writer.add_summary(summary,iIter)
		avgLoss = lossPerIm / batchSizeTrain
		print("Iteration " + str(iIter) + ", train loss = " + str(avgLoss))

	if saveInterval > 0 and (iIter % saveInterval == 0 or iIter == (nIters-1)) :
		outFile = outDir + os.path.sep + baseOutName + '_iter' + str(iIter) + '.ckpt'
		save_path = saver.save(sess,outFile)
		print("Model checkpoint saved to file: " + save_path)


endTime = time.time();
print("Finished optimization. Elapsed time: ")
print(endTime-startTime)

## ----- testing ---- 

dispMaps = forward.eval(feed_dict={data:trainImBatch})
inIms = data_t1.eval(feed_dict={data:trainImBatch})
outIms = data_t2.eval(feed_dict={data:trainImBatch})
predIms = data_t1_prime.eval(feed_dict={data:trainImBatch})

ofn.show_results(trainImBatch,dispMaps,0)


# nx = 2
# ny = 2

# v = np.array([
#   [0.2, 0.2],
#   [0.4, 0.4],
#   [0.6, 0.6],
#   [0.8, 0.8]])

# p = tf.constant(v.reshape([1, nx*ny, 2]), dtype=tf.float32)


# iIm= 0;
# im1 = trainIms[0][:,:,iIm,:].copy()

# shape = (1,) + im1.shape + (1,)

# p = tf.constant(v.reshape([1, nx*ny, 2]), dtype=tf.float32)
# t_img = tf.constant(im1.reshape(shape), dtype=tf.float32)
# t_img = TPS_STN(t_img, nx, ny, p, im1.shape)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())










