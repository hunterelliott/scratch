import tensorflow as tf
from keras.layers import Conv2D, Input, Conv2DTranspose
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import matplotlib.animation as animation

def transition_model(input_shape):

    x = Input(shape=input_shape)

    transition_dim = 16
    kernel_size = 5
    stride = 2
    h = Conv2D(transition_dim,kernel_size,strides=(stride,stride),padding='same',activation='tanh',name='encode_conv')(x)
    #h = Conv2D(transition_dim, 5, padding='same', activation='tanh', name='encode_conv')(h)
    if stride == 1:
        x_prime = Conv2D(input_shape[2], kernel_size, padding='same', activation='tanh', name='decode_conv')(h)
    else:
        x_prime = Conv2DTranspose(input_shape[2], kernel_size, strides=(stride,stride), padding='same', activation='tanh', name='decode_conv')(h)


    model = Model(inputs=x, outputs=x_prime)

    return model

def iteration_model(C,n_t,input_shape):

    X_t_0 = Input(shape=input_shape)
    X_t_prime = C(X_t_0)
    for t in range(n_t-1):
        X_t_prime = C(X_t_prime)

    model = Model(inputs=X_t_0,outputs=X_t_prime)

    return model

def get_X_t(t, R, X_t_0):


    ts = [R.nodes_by_depth[depth][0].output_tensors[0] for depth in R.nodes_by_depth if
          R.nodes_by_depth[depth][0].output_tensors[0].name.find('decode') >= 0]
    n_t = len(ts)
    C_intermediate = K.function([R.layers[0].input], [ts[n_t - t - 1]])
    X_t = np.squeeze(C_intermediate([X_t_0])[0])
    return X_t

def conservation_term(X_t_0,X_t_1):

    #Absolute rate of change of mass
    dMdt = tf.abs(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(X_t_0, 3), 2), 1) -
                  tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(X_t_1, 3), 2), 1))


    return dMdt

def get_train_X(X_shape,n_X_train):

    n_particle_per_t = 3
    thresh = 1-n_particle_per_t/np.prod(X_shape)
    X_t_0_train = (np.random.uniform(0,1,(n_X_train,) + X_shape) > thresh).astype(np.float32)

    return X_t_0_train

def generate_train_X(X_shape,batch_size):

    while True:
        X = get_train_X(X_shape,batch_size)
        yield (X,X)


def scale_im(X,contrast=1.0):

    if contrast > 0:
        X = X * contrast
    else:
        X = np.tanh(X * contrast)

    X = (X + 1) / 2
    return X

def update_anim(t,contrast,im_han,*args):

    im_han.set_array(scale_im(get_X_t(t,*args),contrast))

