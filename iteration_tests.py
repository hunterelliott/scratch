import tensorflow as tf
from keras.layers import Conv2D, Input
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import matplotlib.animation as animation

import iterations as I

# --- Parameters --- #

# Model params
n_t_transition = 1
n_t = 32
im_shape = (256,256,3)

#Optimizer params
lr = 5e-7
lr_decay = 1e-3
momentum = 0.5
n_epochs = int(5)

# ---- Initialization --- #

C = I.transition_model(n_t_transition,im_shape)


#X_t0 = np.random.normal(0,1,(1,)+im_shape)
X_t0 = np.zeros((1,) + im_shape)
X_t0[0,128,128,:] = 1

n_X_train = 8192
n_particle_per_t = 3
thresh = 1-n_particle_per_t/np.prod(im_shape)
X_t_0_train = (np.random.uniform(0,1,(n_X_train,) + im_shape) > thresh).astype(np.float32)


# ---- Optimization ---- #

# Optimize the transition model
#optimizer = optimizers.SGD(lr=lr,decay=lr_decay,momentum=momentum)
optimizer = optimizers.rmsprop(lr=lr)
C.compile(loss=I.conservation_term,optimizer=optimizer)

C.fit(x=X_t_0_train,y=X_t_0_train,epochs=n_epochs,batch_size=256)

# ---- Iteration ----- #

R = I.iteration_model(C,n_t,im_shape)


# ----- Visualization ---- ##

fig = plt.figure()
t_0 = 0
im_han = plt.imshow(I.get_X_t(t_0,R,X_t0))

ani = animation.FuncAnimation(fig,I.update_anim,fargs=(im_han,R,X_t0),frames=n_t,interval = 256)


plt.show()

jkl=1
#TODO - define objective function for ||dXdt|| as tf tensor, use as custom loss

