
import numpy as np

'''Routines for reading the CIFAR-10 python batch files.'''

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
TRAINSET_NAME_LIST = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
TESTSET_NAME = 'test_batch'


def load_cifar10(path):
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
    train_images: in raw format, [None, 3024],
    train_labels: in one-hot format,
    test_images: the same as train_images,
    test_labels: the same as train_labels.
    '''
    _train_batch_files = []
    for name in TRAINSET_NAME_LIST:
        _train_batch_files.append('{}/{}'.format(path, name))
    test_batch = '{}/{}'.format(path, TESTSET_NAME)

    import cPickle

    _train_batches = []
    for batch_file in _train_batch_files:
        with open(batch_file, 'rb') as f:
            _train_batches.append(cPickle.load(f))

    with open(test_batch, 'rb') as f:
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
