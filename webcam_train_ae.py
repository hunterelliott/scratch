import cv2
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras import backend as K




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

cv2.imshow(in_window,im)


pred_window = "prediction"
cv2.namedWindow(pred_window,cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(pred_window,600,600)




AE = Sequential()

# n_coder_layers = 8
# layer_base_dim = 4
n_coder_layers = 2
layer_base_dim = 8


AE.add(Conv2D(im_shape[-1],3,strides=(2,2),input_shape=im_shape))

layer_dims = [layer_base_dim * 2 ** i_layer for i_layer in range(n_coder_layers)]

# Encoder
for layer_dim in layer_dims:
    AE.add(Conv2D(layer_dim,3,strides=(2,2),padding='same'))
# Decoder
for layer_dim in reversed([3] + layer_dims):
    AE.add(Conv2DTranspose(layer_dim,3,strides=(2,2),padding='same'))

#rmsprop works well for <=8 layers and < 8 base dim
#AE.compile(optimizer='rmsprop',loss='mean_squared_error')
#adam works well with 8 layers 8 base dim
AE.compile(optimizer='adam',loss='mean_squared_error')

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


i_iter = 0

def webcam_im_generator(model):
    global graph
    keep_fuckin_goin = True
    i_iter = 0
    while keep_fuckin_goin:
        keep_fuckin_goin,im = vc.read()
        cv2.imshow(in_window, im)

        key = cv2.waitKey(20)
        im1 = pre_process_im(im)

        for t in range(10):
            keep_fuckin_goin, im = vc.read()
        im2 = pre_process_im(im)

        with graph.as_default():

            im_pred = model.predict([im2])
            im_pred = de_process_im(im_pred)
            cv2.imshow(pred_window,im_pred)
        i_iter = i_iter + 1

        if key == 27:
            break

        yield (im1,im2)


AE.fit_generator(webcam_im_generator(AE),steps_per_epoch=10,epochs=1e3)
jkl=1

