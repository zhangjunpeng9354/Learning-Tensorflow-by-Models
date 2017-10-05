import os

import tensorflow as tf
import numpy as np

'''Explaining how to custom input for tensorflow models.


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
    sess = tf.Session()

    dataset1 = tf.contrib.data.Dataset.from_tensors(tf.random_uniform([4, 10]))
    print(dataset1.output_types)
    print(dataset1.output_shapes)

    dataset2 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    print(dataset2.output_types)
    print(dataset2.output_shapes)

    dataset3 = tf.contrib.data.Dataset.from_tensors(
        (tf.random_uniform([4, 2], name='labels'),
         tf.random_uniform([4, 10], name='features'))
    )
    print(dataset3.output_types)
    print(dataset3.output_shapes)

    dataset4 = tf.contrib.data.Dataset.from_tensor_slices(
        (tf.random_uniform([4,2], name='labels'),
         tf.random_uniform([4, 2000], name='features'))
    )
    print(dataset4.output_types)
    print(dataset4.output_shapes)


    dataset5 = tf.contrib.data.Dataset.range(100)
    print(dataset5)
    iterator = dataset5.make_one_shot_iterator()
    next_element = iterator.get_next()

    for i in range(10):
        value = sess.run(next_element)
        print(value)

    iterator = dataset2.make_initializable_iterator()
    next_element = iterator.get_next()

    sess.run(iterator.initializer)
    for i in range(4):
        value = sess.run(next_element)
        print(value)


    dataset6 = tf.contrib.data.Dataset.from_tensor_slices(
        tf.random_uniform([200, 2], name='features')
    )
    print(dataset6.output_types)
    print(dataset6.output_shapes)

    batch1 = dataset6.batch(4)
    print(batch1.output_types)
    print(batch1.output_shapes)

    iterator = batch1.make_initializable_iterator()
    next_element = iterator.get_next()

    sess.run(iterator.initializer)
    for i in range(5):
        value = sess.run(next_element)
        print(value)



    # feeding()
    # input_pipeline('/Users/Zhang/Research/Deep Learning Dataset/CIFAR/cifar-10-batches-py')
