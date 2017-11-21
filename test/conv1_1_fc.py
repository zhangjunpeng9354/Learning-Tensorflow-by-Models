import math

import numpy as np
import tensorflow as tf

# kernel_shape = [1, 1, 3, 2]
# stride = [1, 1, 1, 1]
# data_format = 'NCHW'
#
# input_image = tf.placeholder(dtype=tf.float32)
# data_format = 'NHWC'
# input_image = tf.transpose(input_image, [0, 2, 3, 1])
#
# kernel = tf.get_variable('W', shape=kernel_shape,
#                          initializer=tf.truncated_normal_initializer(
#                              stddev=math.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[3])),
#                              dtype=tf.float32),
#                          dtype=tf.float32)
# bias = tf.get_variable('b', kernel_shape[3], initializer=tf.constant_initializer(0.001))
#
# conv = tf.nn.conv2d(input_image, kernel, stride, 'SAME', data_format=data_format)
# conv = tf.nn.bias_add(conv, bias, data_format=data_format)
#
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# x = np.ones(shape=[1, 4, 4, 3], dtype=np.float32)
# print('x = \t{}'.format(x))
#
# shape = tf.shape(x)
#
# y = sess.run(tf.argmax(tf.reshape(conv, [shape[1], shape[2], 2]), axis=2), feed_dict={input_image: x})
# # y = sess.run(tf.argmax(tf.reshape(conv, [-1, 2]), axis=1), feed_dict={input_image: x})
#
# # y = np.reshape(y, [4, 4])
# print('y = \t{}'.format(y))
#
# print(np.reshape(np.ones([2,2]), [-1]))

# dims = 8
# pos = np.random.randint(0, dims - 1)
#
# logits = tf.random_uniform([dims], maxval=3, dtype=tf.float32)
# labels = tf.one_hot(pos, dims)
#
# res1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
# res2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.constant(pos))
#
# c = tf.reshape(labels, [-1, 2])
# d = tf.reshape(c, [-1])
#
# with tf.Session() as sess:
#     _logits, _labels, _c, _d = sess.run([logits, labels, c, d])
#     print('logits:\t {}'.format(_logits))
#     print('labels:\t {}'.format(_labels))
#     print _c
#     print _d
#     a, b = sess.run([res1, res2])
#     print a, b
#     print a == b


loss = tf.random_uniform([6,5], maxval=3, dtype=tf.float32)
cost = tf.constant([0.125, 0.125, 0.5, 0.125, 0.125])

weight_loss = tf.multiply(cost,loss)

with tf.Session() as sess:
    _loss, _cost, _weight_loss = sess.run([loss, cost, weight_loss])
    print _loss
    print _cost
    print _weight_loss