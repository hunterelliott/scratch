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

n_t = 32

im_shape = (256,256,3)

C = transition_model(n_t,im_shape)

#im = np.random.normal(0,1,(1,)+im_shape)
im = np.zeros((1,) + im_shape)
im[0,128,128,:] = 1


im2 = C.predict(im)


plt.imshow(np.squeeze(im2))

fig = plt.figure(2)



def get_t(t):
    global C,im,n_t
    ts = [C.nodes_by_depth[depth][0].output_tensors[0] for depth in C.nodes_by_depth if
          C.nodes_by_depth[depth][0].output_tensors[0].name.find('decode') >= 0]
    C_intermediate = K.function([C.layers[0].input], [ts[n_t-t-1]])
    im_int = np.squeeze(C_intermediate([im])[0])
    return im_int

anim_i = 0
im_han = plt.imshow(get_t(anim_i))

def update_anim(*args):

    global anim_i
    anim_i += 1
    im_han.set_array(get_t(anim_i))


ani = animation.FuncAnimation(fig,update_anim,frames=31,interval = 250)

plt.show()
jk=1