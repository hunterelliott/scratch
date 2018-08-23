from igan.models import build_discriminator, build_generator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model


G = build_generator()
D = build_discriminator()


d_on_g = Model(inputs=G.input,outputs=D(G.output))
d_on_x = D


# TODO - scaling, augmentation, etc!
x_gen = ImageDataGenerator()


j=1


