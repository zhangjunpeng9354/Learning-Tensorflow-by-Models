import os

import tensorflow as tf
import numpy as np

'''Routines for reading the CIFAR-10 python batch files.'''

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

DEFALFT_DATASET_DIR = '/datasets/cifar-10-batches-py'


def load_cifar10(path=DEFALFT_DATASET_DIR):
    '''Read training and testing samples from CIFAR-10 python batch files.

    Recommendation: AlexNet in the tutorials provided by TensorFlow describes
    a routine for reading CIFAR-10 binary file. See details at
    https://github.com/tensorflow/models/tree/master/tutorials/image.

    No parallelism is supported since the data is small.

    :param path: path to the folder that contains the downloaded datasets.
    For example, if the downloaded file are extracted to '/tmp/CIFAR-10',
    then path should be '/tmp/CIFAR-10/cifar-10-batches-py'.
    Please note that the dataset should be downloaded in advance.
    :return:
    train_images: in raw format, [None, 3024], dtype = np.float32.
    train_labels: in one-hot format, dtype = np.int32.
    test_images: the same as train_images.
    test_labels: the same as train_labels.
    '''

    _training_file_names = [os.path.join(path, 'data_batch_%d' % i) for i in xrange(1, 6)]
    _test_file_names = os.path.join(path, 'test_batch')



    import cPickle

    _train_batches = []
    for _file_name in _training_file_names:
        with open(_file_name, 'rb') as _training_file:
            _train_batches.append(cPickle.load(_training_file))

    with open(_test_file_names, 'rb') as f:
        test_batch = cPickle.load(f)

    train_images = np.vstack(batch['data'] for batch in _train_batches)
    train_ys = np.hstack(batch['labels'] for batch in _train_batches)

    test_images = test_batch['data']
    test_ys = np.array(test_batch['labels'])

    train_labels = np.zeros(shape=(len(train_ys), 10), dtype=np.float32)
    train_labels[np.arange(len(train_ys)), train_ys] = 1
    test_labels = np.zeros(shape=(len(test_ys), 10), dtype=np.float32)
    test_labels[np.arange(len(test_ys)), test_ys] = 1

    return train_images, train_labels, test_images, test_labels
