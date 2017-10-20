from time import time
import math

import numpy as np

'''Building the CIFAR-10 VGG-Net

Summary of available functions:



'''

CACHE_DIR = '/home/ubuntu/notebook/tensorboard/vgg-cifar10'


# CACHE_DIR = '/Users/Zhang/Research/Programming/Learning-Tensorflow-by-Models'


def load_cifar10(path=None):
    if path is None:
        DATASET_DIR = '/Users/Zhang/Research/Deep Learning Dataset/CIFAR/cifar-10-batches-py/'
        # DATASET_DIR = '/home/ubuntu/datasets/cifar-10-batches-py/'
    else:
        DATASET_DIR = path

    TRAINSET_NAME = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    # TRAINSET_NAME = ['data_batch_1']
    TESTSET_NAME = 'test_batch'

    _train_batch_files = []
    for name in TRAINSET_NAME:
        _train_batch_files.append('{}{}'.format(DATASET_DIR, name))
    test_batch = '{}{}'.format(DATASET_DIR, TESTSET_NAME)

    import cPickle

    _train_batches = []
    for batch_file in _train_batch_files:
        with open(batch_file, 'rb') as f:
            _train_batches.append(cPickle.load(f))

    train_images = np.vstack(batch['data'] for batch in _train_batches)

    train_images = train_images.astype(dtype=np.float32)

    R = train_images[:, 0:1024]
    G = train_images[:, 1024:2048]
    B = train_images[:, 2048:]

    R_mean = np.mean(R)
    G_mean = np.mean(G)
    B_mean = np.mean(B)

    R_std = np.std(R)
    G_std = np.std(G)
    B_std = np.std(B)

    return R_mean, G_mean, B_mean, R_std, G_std, B_std


if __name__ == '__main__':
    # load data
    R_mean, G_mean, B_mean, R_std, G_std, B_std = load_cifar10()
    print('{}\t{}\t{}\n{}\t{}\t{}'.format(R_mean, G_mean, B_mean, R_std, G_std, B_std))
