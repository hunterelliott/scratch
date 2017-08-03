import os

import tensorflow as tf
from keras.layers import Conv2D, Input
from keras.models import Model, load_model
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
#lr_decay = 1e-10
lr_decay = 0
momentum = .5
batch_size = 32
n_epochs = 1e1
n_batch_per_epoch = 8

#Visualization params
contrast = -1000


# ---- Initialization --- #

#Create X_0 for visualization
#X_t0 = np.random.normal(0,1,(1,)+im_shape)
X_t0 = np.zeros((1,) + im_shape)
X_t0[0,128,128,:] = 1
X_t0[0,160,160,0] = 1

restore = True

if not restore:

    C = I.transition_model(im_shape)
    R = I.iteration_model(C, n_t, im_shape)

    #optimizer = optimizers.SGD(lr=lr,decay=lr_decay,momentum=momentum)
    optimizer = optimizers.Adam(lr=lr,decay=lr_decay)
    #optimizer = optimizers.rmsprop(lr=lr,decay=lr_decay)
    R.compile(loss=I.conservation_term,optimizer=optimizer)
    cb = [callbacks.TensorBoard(log_dir='/home/hunter/Desktop/TEMP/tmp_tensorboard1',write_graph=False)]
    cb = cb + [callbacks.ReduceLROnPlateau(monitor='loss',factor=.5,patience=2)]


    # ---- Optimization ---- #

    # Optimize the transition model

    R.fit_generator(I.generate_train_X(im_shape, batch_size), n_batch_per_epoch, epochs=n_epochs, callbacks=cb,max_q_size=128)
    #C.fit_generator(I.generate_train_X(im_shape,batch_size),n_batch_per_epoch,epochs=n_epochs,callbacks=cb,max_q_size=128)
else:

    #C = load_model('/media/hunter/1E52113152110F61/shared/Training_Experiments/Iteration/Transition_Models/v0_h16dim_conservation/model0.h5',custom_objects={'conservation_term':I.conservation_term})
    #R = load_model('/media/hunter/1E52113152110F61/shared/Training_Experiments/Iteration/Iteration_Models/v0_h32dim_nt32_conservation/model0.h5',custom_objects={'conservation_term':I.conservation_term})
    R = load_model('/media/hunter/1E52113152110F61/shared/Training_Experiments/Iteration/Iteration_Models/v0_h16dim_nt32_conservation/model0.h5',custom_objects={'conservation_term':I.conservation_term})


# ---- Iteration ----- #


#R.fit_generator(I.generate_train_X(im_shape,batch_size),n_batch_per_epoch,epochs=n_epochs,callbacks=cb,max_q_size=128)

# ----- Visualization ---- ##

fig = plt.figure()
t_0 = 0
im_han = plt.imshow(I.scale_im(I.get_X_t(t_0,R,X_t0),contrast) ,clim=(-1, 1))
ani = animation.FuncAnimation(fig,I.update_anim,fargs=(contrast,im_han,R,X_t0),frames=n_t,interval = 250)

# ---- Trend visualization --- #

tot_mass = np.zeros(n_t)

for t in range(n_t):

    X_t = I.get_X_t(t,R,X_t0)
    tot_mass[t] = np.sum(X_t)

fig = plt.figure(2)
plt.plot(tot_mass)

#C.save('/media/hunter/1E52113152110F61/shared/Training_Experiments/Iteration/Transition_Models/v0_h16dim_conservation/model0.h5')
#R.save('/media/hunter/1E52113152110F61/shared/Training_Experiments/Iteration/Iteration_Models/v0_h16dim_nt32_conservation/model0.h5')
#R.save('/media/hunter/1E52113152110F61/shared/Training_Experiments/Iteration/Iteration_Models/v0_h32dim_nt32_conservation/model0.h5')

plt.show()
jkl=1
#TODO - define objective function for ||dXdt|| as tf tensor, use as custom loss

