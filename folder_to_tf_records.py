import os
import argparse
import time
import numpy as np

import tensorflow as tf
from scipy import misc

parser = argparse.ArgumentParser()
parser.add_argument('directory', metavar='dir', type=str,
                    help='a direcory containing image files to convert to a tfrecords file')

parser.add_argument('output', metavar='out', type=str,
                    help='name for the resulting tfrecords file')

args = parser.parse_args()

d = args.directory;

ims = os.listdir(d)
nIms = len(ims)

print("Converting folder " + d)
print("Found " + str(nIms) + " images.")

startTime = time.time();

#Keep tensorflow from placing anything on GPU
os.environ["CUDA_VISIBLE_DEVICES"]=''

with tf.device('/cpu:0'), tf.Session() as sess:


	writer_opts = tf.python_io.TFRecordOptions(2)#Set compression to zlib
	writer = tf.python_io.TFRecordWriter(args.output,options=writer_opts)

	for i in range(5000):


		#Read the image and encode to jpeg. 
		im = misc.imread(d + os.path.sep + ims[i]);
		im_raw_jpg = im.astype(np.int8).tostring()
		#im_raw_jpg = sess.run(tf.image.encode_jpeg(im));

		#Create tf example and write to tfrecords file
		example = tf.train.Example(features=tf.train.Features(feature={
			'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[im.shape[0]])),
			'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[im.shape[1]])),
			'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[im.shape[2]])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_raw_jpg]))}))

		writer.write(example.SerializeToString())

		if i%50 == 0:
			print("Finished image " + str(i) + " of " + str(nIms))

	writer.close()
endTime = time.time();
print("Finished conversion. Elapsed time: " + str(endTime-startTime) + " seconds.")
