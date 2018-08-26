import os
import cv2
import time
import random
import numpy as np
import time
#import queue

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras import backend as K
from multiprocessing import Process, Manager, Queue, Lock

from itertools import combinations



#cv2.startWindowThread()
in_window = "input"
cv2.namedWindow(in_window,cv2.WINDOW_KEEPRATIO)
win_size = (600,600,3)
cv2.resizeWindow(in_window,win_size[0],win_size[1])

vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, im= vc.read()
else:
    rval = False

#im_shape = (256,256,3)
im_shape = (512,512,3)
#Number of sequential frames to predict from
seq_len = 1
input_shape = (im_shape[0],im_shape[1],im_shape[2]*seq_len)

cv2.imshow(in_window,im)


pred_window = "prediction"
cv2.namedWindow(pred_window,cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(pred_window,600,600)


#If > 0 then train on images where input is missing a portion this large from center
in_painting_size = 0

batch_size = 8

def build_model():

    AE = Sequential()

    n_coder_layers = 2
    layer_base_dim = 6
    kernel_size = 5
    #n_coder_layers = 3
    #layer_base_dim = 8

    #n_coder_layers = 4
    #layer_base_dim = 8

    randomize_queue = True


    assert not (randomize_queue and seq_len>1), "Randomizing queue with sequences is weird"

    layer_dims = [layer_base_dim * 2 ** i_layer for i_layer in range(n_coder_layers)]

    # Encoder

    AE.add(Conv2D(layer_dims[0],kernel_size,strides=(2,2), padding='same', input_shape=input_shape, activation='relu'))
    for layer_dim in layer_dims[1::]:
        AE.add(Conv2D(layer_dim,kernel_size,strides=(2,2), padding='same', activation='relu'))
    # Decoder
    for layer_dim in reversed(layer_dims[0:-1]):
        AE.add(Conv2DTranspose(layer_dim,kernel_size,strides=(2,2), padding='same', activation='relu'))

    # Match input domain
    AE.add(Conv2DTranspose(im_shape[-1],kernel_size, strides=(2,2), padding='same', activation='tanh'))


    #rmsprop works well for <=8 layers and < 8 base dim
    #AE.compile(optimizer='rmsprop',loss='mean_squared_error')
    #adam works well with 8 layers 8 base dim
    AE.compile(optimizer='adam',loss='mean_squared_error')

    #AE.compile(optimizer='sgd',loss='mean_squared_error')

    AE._make_predict_function()

    return AE
#    graph = K.get_session().graph

assert seq_len == 1, "I broke it and I don't care right now"

#Define image pre-processing function
def pre_process_im(im):
    im = cv2.resize(im.copy(),im_shape[0:2])
    im = (im.reshape((1,) +  im_shape).astype(np.float32) / 127.5) - 1
    return im

def de_process_im(im):
    #im = cv2.resize(im.copy(),win_size[0:2])
    im = np.squeeze(im)
    im = (im + 1) * 127.5
    im[im>255] = 255
    im[im<0] = 0
    return im.astype(np.uint8)

#Setup the queue for storing images
#maxQ = 2 *  seq_len
maxQ = 4096
#q = Queue(maxsize=maxQ)

i_iter = 0

manager = Manager()
pool = manager.list([])
pred_pool = manager.list([])

def acquire_to_list(list):

    acquiring=True
    while acquiring:

        acquiring, image = vc.read()
        image = pre_process_im(image)
        list.append((image, []))

        if len(list) > maxQ:
            list.pop(0)


def display_from_list(list, in_window, pred_window):

    key = 0
    while not(key==27):
        if len(list) > 0:
            curr_pair = list[-1]
            if len(curr_pair[1])==0:
                # Wait until we have a prediction
                time.sleep(.05)
            else:
                cv2.imshow(in_window, de_process_im(curr_pair[0][0, :]))
                cv2.imshow(pred_window, de_process_im(curr_pair[1][0,:]))
                print("displaying prediction")
                key = cv2.waitKey(5)


# def predict_from_list(list, window, model):
#
#     key = 0

#         while not(key==27):
#             if len(list)>0:
#
#                 im_pred = list[-1]
#                 with image_lock:
#                     cv2.imshow(window, de_process_im(im_pred[0,:]))
#                     key = cv2.waitKey(5)

def train_on_list(list, batch_size):

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("Initializing training process...")

    model = build_model()

    i_iter = 0
    while True:
        while len(list) < batch_size:
            time.sleep(.05)
        minibatch_ind = [random.randint(0,min(maxQ,len(list))-1) for _ in range(batch_size)]
        im_batch = np.concatenate([list[ind][0] for ind in minibatch_ind], axis=0)

        model.train_on_batch(im_batch, im_batch)
        i_iter += 1
        # Update the prediction for the current frame
        curr_input = list[-1][0]
        curr_pair = (curr_input, model.predict(curr_input))
        # We may occasionally overwrite latest acquisition, that's OK
        list[-1] = curr_pair

        if i_iter % 10 == 0:
            print(i_iter)






all_proc = []
all_proc.append(Process(target=acquire_to_list, args=(pool,)))
all_proc.append(Process(target=display_from_list, args=(pool, in_window, pred_window)))
all_proc.append(Process(target=train_on_list, args=(pool, batch_size)))
#pred_proc = Process(target=predict_from_list, args=(pool, pred_window, AE))


[proc.start() for proc in all_proc]
#AE.fit_generator(generate_train_from_list(pool, batch_size),steps_per_epoch=maxQ,epochs=5e3,max_queue_size=128)

[proc.join() for proc in all_proc]

j=1
