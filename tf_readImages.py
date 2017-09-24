import os
import numpy as np
from scipy import misc

def preProcIm(im):

		#im = misc.imresize(im,(256,256)) #TEMP!!
		im = misc.imresize(im,(32,32)) #TEMP!!

		im = im.astype(float)/255
		if np.ndim(im) == 2:
			im = np.expand_dims(im,2)

		return im	


def readAllClassImages(trainParentDir,testParentDir):

	#Reads all images from all classes in the specified directory. Each sub-directotry is assumed to contain images from a class and the class name is the name of that directory

	#Find class sub-directories. We assume same structure in test and train
	classNames = os.listdir(trainParentDir)
	nClasses = len(classNames)

	print('Found ' + str(nClasses) + ' class folders:')
	print(classNames)

	nTrainPerClass = np.zeros(nClasses,dtype=np.int)
	nTestPerClass = np.zeros(nClasses,dtype=np.int)

	trainFiles = [ [] for _ in range(nClasses)]
	testFiles = [ [] for _ in range(nClasses)]

	trainIms =  [ [] for _ in range(nClasses)]
	testIms =  [ [] for _ in range(nClasses)]

	trainLabels = [ [] for _ in range(nClasses)]
	testLabels = [ [] for _ in range(nClasses)]

	for iClass in range(0, nClasses):

		#Get all file names and count images
		currTrainDir = trainParentDir + os.path.sep + classNames[iClass]
		trainFiles[iClass] = os.listdir(currTrainDir)
		nTrainPerClass[iClass] = int(len(trainFiles[iClass]))		

		currTestDir = testParentDir + os.path.sep + classNames[iClass]
		testFiles[iClass] = os.listdir(currTestDir)
		nTestPerClass[iClass] = len(testFiles[iClass])

		#Read one iamge to size arrays
		tmp = preProcIm(misc.imread(currTrainDir + os.path.sep + trainFiles[iClass][0]))

		imSize = tmp.shape[0:2]
		if tmp.ndim > 2:
			nImChan = tmp.shape[2]
		else:
			nImChan = 1;


		print("Image size:" + str(imSize))

		trainIms[iClass] = np.zeros(imSize + (nTrainPerClass[iClass], nImChan)) #Use a list of np arrays to allow for varying numbers of iamges per class 
		testIms[iClass] = np.zeros(imSize + (nTestPerClass[iClass],nImChan))

		print("Loading images for class " + str(iClass))
		for iIm in range(0,nTrainPerClass[iClass]):
			trainIms[iClass][:,:,iIm,:] = preProcIm(misc.imread(currTrainDir + os.path.sep + trainFiles[iClass][iIm]))
			if iIm%500 == 0:
				print("Loaded image " + str(iIm) + " of " + str(nTrainPerClass[iClass]))
		for iIm in range(0,nTestPerClass[iClass]):
			testIms[iClass][:,:,iIm] = preProcIm(misc.imread(currTestDir + os.path.sep + testFiles[iClass][iIm]))


		#Setup labels	
		trainLabels[iClass] = np.zeros((nTrainPerClass[iClass],nClasses),dtype=np.bool)
		trainLabels[iClass][:,iClass] = True
		testLabels[iClass] = np.zeros((nTestPerClass[iClass],nClasses),dtype=np.bool)
		testLabels[iClass][:,iClass] = True

	return trainIms, testIms, trainLabels, testLabels

def readAllTimeSeriesImages(trainParentDir,testParentDir):

	#Reads all images from all of the timeseries in each folder classes in the specified directory. Each sub-directotry is assumed to contain an image series, with one image per timepoint.

	#Find time series sub-directories
	trainTSNames = os.listdir(trainParentDir)
	testTSNames = os.listdir(testParentDir)
	nTSTrain = len(trainTSNames)
	nTSTest = len(testTSNames)

	print('Found ' + str(trainTSNames) + ' train time series folders:')
	print(trainTSNames)

	nTrainPerTS = np.zeros(nTSTrain,dtype=np.int)
	nTestPerTS = np.zeros(nTSTest,dtype=np.int)

	trainFiles = [ [] for _ in range(nTSTrain)]
	testFiles = [ [] for _ in range(nTSTest)]

	trainIms =  [ [] for _ in range(nTSTrain)]
	testIms =  [ [] for _ in range(nTSTest)]

	for iTS in range(0, nTSTrain):

		#Get all file names and count images
		currTrainDir = trainParentDir + os.path.sep + trainTSNames[iTS]
		trainFiles[iTS] = sorted(os.listdir(currTrainDir))
		nTrainPerTS[iTS] = int(len(trainFiles[iTS]))		

		#Read one image to size arrays
		tmp = preProcIm(misc.imread(currTrainDir + os.path.sep + trainFiles[iTS][0]))

		imSize = tmp.shape[0:2]
		if tmp.ndim > 2:
			nImChan = tmp.shape[2]
		else:
			nImChan = 1;


		print("Image size:" + str(imSize))

		trainIms[iTS] = np.zeros(imSize + (nTrainPerTS[iTS], nImChan)) #Use a list of np arrays to allow for varying numbers of iamges per class 		

		print("Loading images for train series " + str(iTS))
		for iIm in range(0,nTrainPerTS[iTS]):
			trainIms[iTS][:,:,iIm,:] = preProcIm(misc.imread(currTrainDir + os.path.sep + trainFiles[iTS][iIm]))
			if iIm%500 == 0:
				print("Loaded image " + str(iIm) + " of " + str(nTrainPerTS[iTS]))
		
	for iTS in range(0, nTSTest):

		#Get all file names and count images
		currTestDir = testParentDir + os.path.sep + testTSNames[iTS]
		testFiles[iTS] = sorted(os.listdir(currTestDir))
		nTestPerTS[iTS] = len(testFiles[iTS])		

		testIms[iTS] = np.zeros(imSize + (nTestPerTS[iTS],nImChan))

		print("Loading images for test series " + str(iTS))
		for iIm in range(0,nTestPerTS[iTS]):
			testIms[iTS][:,:,iIm] = preProcIm(misc.imread(currTestDir + os.path.sep + testFiles[iTS][iIm]))

		
	return trainIms, testIms