import math

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, InputLayer


def get_layer_dimensions(z_shape=(1,1,128), stride=2):

    return [int(max(z_shape[-1] / 2 ** x, 3)) for x in range(int(math.floor(math.log(z_shape[-1], stride))))]


def build_generator(z_shape=(1, 1, 128), stride=2, activation='relu', kernel_size=3):


    model = Sequential()
    model.name = 'generator'
    model.add(InputLayer(input_shape=z_shape))
    layer_dims = get_layer_dimensions(z_shape=z_shape, stride=stride)


    for i_layer, layer_dim in enumerate(layer_dims):
        model.add(Conv2DTranspose(layer_dim, kernel_size=kernel_size, activation=activation,
                                  strides=stride, padding='same'))

    return model


def build_discriminator(z_shape=(1, 1, 128), stride=2, activation='relu', kernel_size=3):
    model = Sequential()
    model.name = 'discriminator'
    layer_dims = get_layer_dimensions(z_shape=z_shape, stride=stride)
    layer_dims.reverse()
    w = stride ** len(layer_dims)
    model.add(InputLayer(input_shape=(w, w, 3)))

    for i_layer, layer_dim in enumerate(layer_dims):
        model.add(Conv2D(layer_dim, kernel_size=kernel_size, activation=activation,
                         strides=stride, padding='same'))

    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    return model



