from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose

import numpy as np

import misc as misc



def auto_encoder(layer_depths, in_shape):

    model = Sequential()

    for i_encode in range(len(layer_depths)):
        model.add(Conv2D(layer_depths[i_encode],3,strides=2,input_shape=in_shape,activation='tanh'))

    for i_encode in range(len(layer_depths)):
        model.add(Conv2DTranspose(layer_depths[i_encode],3,strides=2,input_shape=in_shape,activation='tanh'))

    return model



n_per_coder = 3

layer_depths = 2 ** np.array(range(n_per_coder))
#in_shape = np.array([256,256,3])
in_shape = [32,32,3]

ae = auto_encoder(layer_depths,in_shape)

data = misc.load_cifar()

jkl=1


