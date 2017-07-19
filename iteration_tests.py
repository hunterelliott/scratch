import tensorflow as tf
from keras.layers import Conv2D, Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import matplotlib.animation as animation

import iterations as I

n_t = 32

im_shape = (256,256,3)

C = I.transition_model(n_t,im_shape)

#X_t0 = np.random.normal(0,1,(1,)+im_shape)
X_t0 = np.zeros((1,) + im_shape)
X_t0[0,128,128,:] = 1




fig = plt.figure()
t_0 = 0
im_han = plt.imshow(I.get_X_t(t_0,C,X_t0))

ani = animation.FuncAnimation(fig,I.update_anim,fargs=(im_han,C,X_t0),frames=31,interval = 250)


plt.show()
jk=1