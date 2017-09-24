import numpy as np
import tensorflow as tf
import os
import tf_readImages as tfr
#from TPS_STN import TPS_STN
import ofnOps as ofn
#from matplotlib import pyplot as plt
import time


# --------- parameters ---------

# -- Model parameters
layerWidth = (16, 32, 64) #Num feature maps per conv layer (order reversed for up-conv layers)
nLayers = len(layerWidth)
maxDisp = 10

# -- Optimization parameters
nIters = int(1e5)
baseLR = 1e-4#Base learning rate
batchSize = 6 #Number of image pairs per image series to include in a batch

# -- logging parameters
logInterval = 100 #Log loss etc every n iterations
saveInterval = int(0) #Save model evern n iterations


# ---------- Input ----------

trainParentDir = '/home/hunter/Desktop/TEMP_LOCAL/Data/MNIST/Train'
testParentDir = '/home/hunter/Desktop/TEMP_LOCAL/Data/MNIST/Test'

outDir = '/home/hunter/Desktop/TEMP_LOCAL/MNIST_AE/Train'
outDirTest = '/home/hunter/Desktop/TEMP_LOCAL/MNIST_AE/Test'
baseOutName = 'MNIST_test' #Base file name for checkpoints (iteration number will be appended)

trainIms,testIms,trainLabels,testLabels = tfr.readAllClassImages(trainParentDir,testParentDir)


nImChan = trainIms[0].shape[3]
#Un-list the data & labels for ease of use below
trainIms = np.concatenate(trainIms,2)
nTrainTot = trainIms.shape[2]
testIms = np.concatenate(testIms,2)
nTestTot =  testIms.shape[2]

trainLabels = np.concatenate(trainLabels[:])
testLabels = np.concatenate(testLabels[:])


# ---------- Model creation ----------


data = tf.placeholder("float",shape=[batchSize,32,32,nImChan],name='data')


# -- down-convolution layers

print('Creating encoding downconvolution layers')
currInput = data
for iLayer in range(0,nLayers):

	currInput = ofn.down_conv(currInput,layerWidth[iLayer],iLayer)

# -- up-convolution layers

print('Creating decoding upconvolution layers')

outLayerWidth = layerWidth[-2::-1] + (nImChan,)
nOutLayers = len(outLayerWidth)

waist = currInput;

for iLayer in range(0,nOutLayers):

	currInput = ofn.up_conv(currInput,outLayerWidth[iLayer],iLayer)

output_AE = currInput

# -- stochastic generative layers

print('Creating stochastic generative layers')

outLayerWidth = layerWidth[-2::-1] + (nImChan,)
nOutLayers = len(outLayerWidth)

waistShape = tf.shape(waist);
#Z = tf.placeholder("float",shape=waistShape)
zSize = (batchSize,4,4,64)

Z = tf.placeholder("float",shape=zSize,name='Z')

#currInput = Z + waist
currInput = Z

for iLayer in range(0,nOutLayers):

	currInput = ofn.up_conv(currInput,outLayerWidth[iLayer],iLayer+nLayers)

output_Gen = currInput

output = output_AE + output_Gen;

loss = tf.nn.l2_loss(data - output,'L2_loss_train')
loss_det = tf.nn.l2_loss(data - output_AE,'L2_loss_train_deterministic')
loss_stoch = tf.nn.l2_loss(data - output_Gen,'L2_loss_train_stochastic')
AE_norm = tf.nn.l2_loss(output_AE,'L2_loss_train_stochastic')
Gen_norm = tf.nn.l2_loss(output_Gen,'L2_loss_train_stochastic')

tf.summary.scalar("loss",loss)
tf.summary.scalar("loss stochastic",loss_stoch)
tf.summary.scalar("loss deterministic",loss_det)
tf.summary.scalar("AE output norm",AE_norm)
tf.summary.scalar("Gen output norm",Gen_norm)

summary_op = tf.summary.merge_all()

## ------- optimization ----------

print("Starting optimization...")
startTime = time.time();	


optimizer = tf.train.AdamOptimizer(baseLR).minimize(loss)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


summary_writer = tf.summary.FileWriter(outDir,graph=tf.get_default_graph())
summary_writer_test = tf.summary.FileWriter(outDirTest,graph=tf.get_default_graph())


with tf.device('cpu:0'):
	saver = tf.train.Saver()


for iIter in range(0,nIters):


	#trainImBatch,testImBatch = ofn.get_image_pair_batch(trainIms,testIms,baseBatchSize)
	iTrainBatch = np.random.randint(0,nTrainTot-1,batchSize)
	iTestBatch = np.random.randint(0,nTestTot-1,batchSize)

	trainBatch = np.transpose(trainIms[:,:,iTrainBatch],(2,0,1,3))
	testBatch = np.transpose(testIms[:,:,iTestBatch],(2,0,1,3))

	trainZBatch = np.random.uniform(-1,1,zSize)		
	testZBatch = np.random.uniform(-1,1,zSize)		


	optimizer.run(feed_dict={data:trainBatch, Z:trainZBatch})
	
	if iIter % logInterval == 0:
		[lossPerIm,summary] = sess.run([loss,summary_op],feed_dict={data:trainBatch, Z:trainZBatch})		
		summary_writer.add_summary(summary,iIter)
		avgLoss = lossPerIm / batchSize
		print("Iteration " + str(iIter) + ", train loss = " + str(avgLoss))

		[lossPerIm,summary] = sess.run([loss,summary_op],feed_dict={data:testBatch, Z:testZBatch})		
		summary_writer_test.add_summary(summary,iIter)
		avgLoss = lossPerIm / batchSize
		print("Iteration " + str(iIter) + ", test loss = " + str(avgLoss))

	if saveInterval > 0 and (iIter % saveInterval == 0 or iIter == (nIters-1)) :
		outFile = outDir + os.path.sep + baseOutName + '_iter' + str(iIter) + '.ckpt'
		save_path = saver.save(sess,outFile)
		print("Model checkpoint saved to file: " + save_path)


endTime = time.time();
print("Finished optimization. Elapsed time: ")
print(endTime-startTime)
