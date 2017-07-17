import pickle
import timeit
import os
import numpy as np

def load_cifar():

    data_dir = '/media/hunter/storage/Data/cifar-10-batches-py'

    all_files = os.listdir(data_dir)

    all_files.sort()

    all_files = [file for file in all_files if file.find('data_batch')>=0]
    all_files = [os.path.join(data_dir,file) for file in all_files]

    im_shape = [32, 32, 3]

    data = np.zeros((0,) + tuple(im_shape), dtype=np.uint8)

    for file in all_files:

        t_start = timeit.default_timer()

        with open(file,'rb') as fid:
            batch = pickle.load(fid, encoding='bytes')
            data = np.concatenate((data, np.reshape(batch["b'data'"], (-1,) + im_shape)), 0)

        t_delta = timeit.default_timer() - t_start

        print("Finished file " + file + " in " + str(t_delta) + " s")

    return data

