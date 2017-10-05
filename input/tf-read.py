import os

import tensorflow as tf
import numpy as np

'''Explaining how to custom input for tensorflow models.

Summary of available available approaches:
1. Feeding: Python code provides the data when running each step,
2. (Recommended) Reading from files: an input pipeline reads the data from files
at the beginning of a TensorFlow graph,
3. (Not Recommended) Pre-loaded data: a constant or variable in the TensorFlow graph
holds all the data (for small data sets).
'''


def feeding():
    x = tf.placeholder(tf.float32, shape=[4, 4], name='x')
    b = tf.constant(1.0, dtype=tf.float32, shape=[4, 4], name='b')
    y = tf.add(x, b, name='y')

    with tf.Session() as sess:
        np_x = np.random.randint(1, high=10, size=[4, 4])

        np_y = sess.run(y, feed_dict={x: np_x})

        print('x = \r {}'.format(np_x))
        print('y = x + 1\r {}'.format(np_y))

    return None


def input_pipeline(data_dir):
    '''A typical pipeline for reading records from files has the following stages:

    1. The list of filenames,
    2. Optional filename shuffling,
    3. Optional epoch limit,
    4. Filename queue,
    5. A Reader for the file format,
    6. A decoder for a record read by the reader,
    7. Optional preprocessing,
    8. Example queue,
    '''

    filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    return None


if __name__ == '__main__':
    feeding()
    input_pipeline('/Users/Zhang/Research/Deep Learning Dataset/CIFAR/cifar-10-batches-py')
