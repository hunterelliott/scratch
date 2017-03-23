
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_yaml

model_file = '/home/hunter/Desktop/TEMP_LOCAL/keras_mnist_test.h5'
model_def = '/home/hunter/Desktop/TEMP_LOCAL/keras_mnist_test_model.yaml'
model_weights = '/home/hunter/Desktop/TEMP_LOCAL/keras_mnist_test_weights.h5'




#with tf.device('/gpu:0'):
with tf.device("/job:ps/task:0/cpu:0"):

	sess = tf.Session()
	K.set_session(sess)
	
	model = load_model(model_file) 