import tensorflow as tf
from keras.layers import Conv2D, Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import matplotlib.animation as animation

def transition_model(n_t,input_shape):

    x = Input(shape=input_shape)

    transition_dim = 16
    encode = Conv2D(transition_dim,5,padding='same',activation='tanh',name='encode_conv')
    decode = Conv2D(input_shape[2], 5, padding='same', activation='tanh',name='decode_conv')

    x_prime = x
    for i in range(n_t):
        h = encode(x_prime)
        x_prime = decode(h)

    model = Model(inputs=x, outputs=x_prime)

    return model

def iteration_model(C,n_t,input_shape):

    X_t_0 = Input(shape=input_shape)
    X_t_prime = C(X_t_0)
    for t in range(n_t-1):
        X_t_prime = C(X_t_prime)

    model = Model(inputs=X_t_0,outputs=X_t_prime)

    return model

def get_X_t(t,C,X_t_0):


    ts = [C.nodes_by_depth[depth][0].output_tensors[0] for depth in C.nodes_by_depth if
          C.nodes_by_depth[depth][0].output_tensors[0].name.find('decode') >= 0]
    n_t = len(ts)
    C_intermediate = K.function([C.layers[0].input], [ts[n_t-t-1]])
    X_t = np.squeeze(C_intermediate([X_t_0])[0])
    return X_t


def update_anim(t,im_han,*args):

    im_han.set_array(get_X_t(t,*args))


def conservation_term(X_t_0,X_t_1):

    #Absolute rate of change of mass
    dMdt = tf.abs(tf.reduce_sum(X_t_0) -  tf.reduce_sum(X_t_1))

    return dMdt