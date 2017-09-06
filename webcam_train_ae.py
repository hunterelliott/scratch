import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

im_shape = (256,256,3)


AE = Sequential()

n_coder_layers = 4
layer_base_dim = 8

AE.add(Conv2D(im_shape[-1],3,strides=(2,2),input_shape=im_shape))

layer_dims = [layer_base_dim * 2 ** i_layer for i_layer in range(n_coder_layers)]

# Encoder
for layer_dim in layer_dims:
    AE.add(Conv2D(layer_dim,3,strides=(2,2),padding='same'))
# Decoder
for layer_dim in reversed([3] + layer_dims):
    AE.add(Conv2DTranspose(layer_dim,3,strides=(2,2),padding='same'))

AE.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

in_window = "input"
cv2.namedWindow(in_window,cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(in_window,600,600)

def webcam_im_generator():
    while True:
        im = vc.read()
        cv2.imshow(in_window, im)
        im = cv2.resize(im,im_shape) - 128.0
        yield (im,im)

AE.fit_generator(webcam_im_generator(),steps_per_epoch=20,epochs=1e2)
jkl=1

