import os

import tensorflow as tf
from keras.layers import Conv2D, Input
from keras.models import Model
from keras import optimizers
from keras import callbacks
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#Optimizer params
lr = 1e-6 #Works OK for unsigned conservation
#lr = 1e-3
lr_decay = 1e-10
lr_decay = 0
momentum = .5
batch_size = 256
n_epochs = 1
n_batch_per_epoch = 4096


# ---- Initialization --- #

C = I.transition_model(im_shape)

#Create X_0 for visualization
#X_t0 = np.random.normal(0,1,(1,)+im_shape)
X_t0 = np.zeros((1,) + im_shape)
X_t0[0,128,128,:] = 1

#optimizer = optimizers.SGD(lr=lr,decay=lr_decay,momentum=momentum)
optimizer = optimizers.Adam(lr=lr,decay=lr_decay)
#optimizer = optimizers.rmsprop(lr=lr,decay=lr_decay)
C.compile(loss=I.conservation_term,optimizer=optimizer)
cb = [callbacks.TensorBoard(log_dir='/home/hunter/Desktop/TEMP/tmp_tensorboard1',write_graph=False)]
#cb = cb + [callbacks.ReduceLROnPlateau(monitor='loss',factor=.75)]


# ---- Optimization ---- #

# Optimize the transition model
C.fit_generator(I.generate_train_X(im_shape,batch_size),n_batch_per_epoch,epochs=n_epochs,callbacks=cb,max_q_size=128)

# ---- Iteration ----- #

R = I.iteration_model(C,n_t,im_shape)


# ----- Visualization ---- ##

fig = plt.figure()
t_0 = 0
im_han = plt.imshow(I.scale_im(I.get_X_t(t_0,R,X_t0)) ,clim=(-1, 1))
plt.colorbar()
ani = animation.FuncAnimation(fig,I.update_anim,fargs=(im_han,R,X_t0),frames=n_t,interval = 250)

# ---- Trend visualization --- #

tot_mass = np.zeros(n_t)

for t in range(n_t):

    X_t = I.get_X_t(t,R,X_t0)
    tot_mass[t] = np.sum(X_t)

fig = plt.figure(2)
plt.plot(tot_mass)



plt.show()
jkl=1
#TODO - define objective function for ||dXdt|| as tf tensor, use as custom loss

