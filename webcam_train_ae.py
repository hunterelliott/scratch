import os
import cv2
import random
import numpy as np
import time
import queue

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras import backend as K

from itertools import combinations

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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


AE = Sequential()

n_coder_layers = 2
layer_base_dim = 6
#n_coder_layers = 3
#layer_base_dim = 8

#n_coder_layers = 4
#layer_base_dim = 8

randomize_queue = True


assert not (randomize_queue and seq_len>1), "Randomizing queue with sequences is weird"

AE.add(Conv2D(im_shape[-1],3,strides=(2,2),input_shape=input_shape))

layer_dims = [layer_base_dim * 2 ** i_layer for i_layer in range(n_coder_layers)]

# Encoder
for layer_dim in layer_dims:
    AE.add(Conv2D(layer_dim,4,strides=(2,2),padding='same'))
# Decoder
for layer_dim in reversed([3] + layer_dims):
    AE.add(Conv2DTranspose(layer_dim,4,strides=(2,2),padding='same'))

#rmsprop works well for <=8 layers and < 8 base dim
#AE.compile(optimizer='rmsprop',loss='mean_squared_error')
#adam works well with 8 layers 8 base dim
#AE.compile(optimizer='adam',loss='mean_squared_error')

AE.compile(optimizer='sgd',loss='mean_squared_error')

AE._make_predict_function()


graph = K.get_session().graph



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
maxQ = 600
q = queue.Queue(maxsize=maxQ)

i_iter = 0

verts = [[] for _ in range(3)]

def webcam_im_generator(model):
    global graph
    keep_fuckin_goin = True

    while keep_fuckin_goin:

        while not q.full():
            keep_fuckin_goin, im = vc.read()
            im_latest = pre_process_im(im)
            q.put(im_latest)


        im1 = np.concatenate([q.queue[i] for i in range(seq_len)],3)
        if randomize_queue:
            random.shuffle(q.queue)
        im2 = q.queue[-1]
        if not randomize_queue:
            im_for_pred = np.concatenate([q.queue[i] for i in range(maxQ-seq_len,maxQ)],3)
        else:
            im_for_pred = im_latest

        oldest_im = q.get() #Keep images moving through the queue
        key = cv2.waitKey(33)

        with graph.as_default():

            im_pred = model.predict([im_for_pred])
            im_pred = de_process_im(im_pred)
            cv2.imshow(in_window, de_process_im(im_for_pred[0,:,:,-im_shape[2]::]))
            cv2.imshow(pred_window,im_pred)


            if key == 27:
                break
            elif key in range(49,52):
                rep = K.function([model.input],[model.layers[n_coder_layers].output])
                verts[key-49] = rep([im_for_pred])[0].flatten()
                print('============= SET VERT ' + str(key-49) + '============')

            if key == 51:
                edge_len = []
                for edge in combinations(range(len(verts)),2):
                    #Edge lengths
                    edge_len.append(np.linalg.norm(verts[edge[0]] - verts[edge[1]]))

                #Give distance of 3 relative to 2
                meanDiff = np.mean(edge_len[1:2])
                relDiff = meanDiff / edge_len[0]
                print('==================' + str(relDiff) + '=============================')

        if in_painting_size > 0:
            i1 = int(input_shape[0] / 2 - in_painting_size)
            i2 = int(input_shape[0] / 2 + in_painting_size)
            im1[:,i1:i2,i1:i2] = 0


        yield (im1,im2)


AE.fit_generator(webcam_im_generator(AE),steps_per_epoch=maxQ,epochs=5e3)
#AE.fit_generator(webcam_im_generator(AE),steps_per_epoch=30,epochs=5e3)
jkl=1

