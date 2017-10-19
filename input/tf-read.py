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

    dataset = tf.contrib.data.Dataset.from_tensor_slices(
        tf.reshape(np.arange(200), [-1, 2])
    )
    print(dataset.output_types)
    print(dataset.output_shapes)

    batch = dataset.batch(4)
    print(batch.output_types)
    print(batch.output_shapes)

    iterator = batch.make_initializable_iterator()
    next_element = iterator.get_next()

    sess.run(iterator.initializer)
    for i in range(5):
        value1, value2 = sess.run([next_element, next_element])
        print(value1, value2)

    print('_' * 35)

    value1, value2 = sess.run([next_element, next_element])
    print(value1, value2)

    value_tensor = tf.Variable(5.0)
    print value_tensor



