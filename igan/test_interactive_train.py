import numpy as np

import cv2
from imutils import build_montages

from igan.models import build_discriminator, build_generator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers, callbacks
from keras import backend as K

# ---

lr = 1e-2
momentum = 0.9
batch_size = 9
n_iters = 100

out_dir = '/home/hunter/Desktop/TEMP/igantests'

# ---

G = build_generator()
D = build_discriminator()

D_on_G = Model(inputs=G.input,outputs=D(G.output))


G._make_predict_function()

def key_code_to_index(code):
    codes = [
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        255 ]
    return codes.index(code)

def get_binary_labels_interactively(X):

    X_norm = X / np.max(X)
    X_norm = (X_norm * 255).astype(np.uint8)
    w = X_norm.shape[1] * 4
    montage = build_montages(X_norm, (w, w) , (3,3) )[0]

    # montage = montage / np.max(montage)
    # montage = (montage * 255).astype(np.uint8)
    cv2.imshow("minibatch", montage)
    key = cv2.waitKey(0)

    Y_ind = np.zeros(X.shape[0], dtype=np.bool)

    Y_ind[key_code_to_index(key)]  = True

    Y = np.zeros((X.shape[0], 2))
    Y[Y_ind, 0] = True
    Y[np.logical_not(Y_ind), 1] = True


    print(key)
    # print(Y)

    return Y

def interactively_label_batch(batch_size=9, Z_shape=(1,1,128)):
    Z = np.random.uniform(0, 1, (batch_size,) + Z_shape)
    # Generate images for labelling
    X = G.predict(Z)
    # Get labels from user
    Y = get_binary_labels_interactively(X)

    return Z, Y, X


def interactive_label_generator(kwargs):

    while True:
        Z, Y, X = interactively_label_batch(**kwargs)
        yield (Z, Y)



optimizer = optimizers.SGD(lr=lr, momentum=momentum)
cb = [callbacks.TensorBoard(log_dir=out_dir, write_graph=True)]


def generator_loss(Y_true, Y_pred):

    return -K.log(Y_pred)


D_on_G.compile(loss=generator_loss, optimizer=optimizer)
D.compile(loss='categorical_crossentropy', optimizer=optimizer)
# D_on_G.fit_generator(interactive_label_generator(batch_size),
#                      steps_per_epoch=1, epochs=1e1,
#                      callbacks=cb)

for i_iter in range(0,n_iters):

    # Get a batch

    Z, Y, X = interactively_label_batch(batch_size)

    # Update D
    d_loss = D.train_on_batch(X, Y)

    # Update G
    g_loss = D_on_G.train_on_batch(Z,Y)

    print("Iteration {}, D loss = {}, G loss = {}".format(i_iter, d_loss, g_loss))


