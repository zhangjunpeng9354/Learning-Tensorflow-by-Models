import math

import tensorflow as tf
import numpy as np


def load_dataset():
    from sklearn.datasets import load_iris

    __dataset = load_iris()
    x = __dataset['data']
    y = __dataset["target"]

    __size = len(x)
    __rand_indices = np.random.permutation(__size)
    x = x[__rand_indices]
    y = y[__rand_indices]

    return x, y


def fc_layer(features, size, name='fc'):
    with tf.variable_scope(name):
        weights = tf.get_variable('w', shape=size, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=math.sqrt(2.0 / (size[0] + size[1])),
                                      dtype=tf.float32)
                                  )

        bias = tf.get_variable('b', shape=size[1], initializer=tf.constant_initializer(0.0001))

        nn = tf.add(tf.matmul(features, weights), bias, name='features')

    return nn


def inference_model(features):
    fc1 = fc_layer(features, [4, 8], name='fc1')
    relu1 = tf.nn.relu(fc1, name='relu1')

    fc2 = fc_layer(relu1, [8, 3], name='fc2')

    return fc2


def loss(logits, labels):
    with tf.variable_scope('loss'):
        __cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        mean_cross_entropy = tf.reduce_mean(__cross_entropy, name='cross_entropy')

    return mean_cross_entropy


def evaluate(logits, labels):
    with tf.variable_scope('accuracy'):
        __preds = tf.argmax(logits, axis=1)
        __correct_preds = tf.equal(__preds, labels)
        acc = tf.reduce_mean(tf.cast(__correct_preds, tf.float32))

    return acc


def train(loss, global_step):
    # __lr = tf.train.exponential_decay(0.001,
    #                                   global_step,
    #                                   5,
    #                                   0.9,
    #                                   staircase=True)
    __lr = 0.01

    __optimizer = tf.train.GradientDescentOptimizer(__lr)
    train_op = __optimizer.minimize(loss, global_step=global_step)

    return train_op


def train_model(x, y):
    samples = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='features')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')

    global_step = tf.Variable(initial_value=0, trainable=False, name='step')

    logits = inference_model(samples)
    cross_entropy_loss = loss(logits, labels)
    accuracy = evaluate(logits, labels)
    train_op = train(cross_entropy_loss, global_step)
    # Train the model and Save it.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

        __global_step = sess.run(global_step)
        while (__global_step < 100):
            # Prepare the data for a train-batch.
            __batch_sample_indices = np.random.randint(len(x), size=24)
            __batch_features = x[__batch_sample_indices]
            __batch_labels = y[__batch_sample_indices]

            __global_step, _ = sess.run([global_step, train_op],
                                        feed_dict={
                                            samples: __batch_features,
                                            labels: __batch_labels
                                        })

            __loss, __accuracy = sess.run([cross_entropy_loss, accuracy],
                                          feed_dict={
                                              samples: __batch_features,
                                              labels: __batch_labels
                                          })

            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f}."
            print(msg.format(__global_step, __accuracy, __loss))
            saver.save(sess, 'model/pretrain_model', global_step=__global_step)


def load_model(x, y):
    # Load the pretrained model.
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/pretrain_model-100.meta')
        saver.restore(sess, tf.train.latest_checkpoint('model/'))

        # Get the placeholder for samples and labels
        graph = tf.get_default_graph()
        samples = graph.get_tensor_by_name('features:0')
        labels = graph.get_tensor_by_name('labels:0')

        logits = graph.get_tensor_by_name('fc2/features:0')
        accuracy = evaluate(logits, labels)
        cross_entropy_loss = graph.get_tensor_by_name('loss/cross_entropy:0')

        global_step = tf.Variable(initial_value=0, trainable=False, name='step')
        train_op = train(cross_entropy_loss, global_step)

        sess.run(global_step.initializer)

        __accuracy, __global_step = sess.run([accuracy, global_step],
                                             feed_dict={
                                                 samples: x,
                                                 labels: y
                                             })

        print('Pretrain accuracy: \t{}.'.format(__accuracy))

        # retrain 100 steps
        while (__global_step < 1000):
            # Prepare the data for a train-batch.
            __batch_sample_indices = np.random.randint(len(x), size=32)
            __batch_features = x[__batch_sample_indices]
            __batch_labels = y[__batch_sample_indices]

            __global_step, _ = sess.run([global_step, train_op],
                                        feed_dict={
                                            samples: __batch_features,
                                            labels: __batch_labels
                                        })

            __loss, __accuracy = sess.run([cross_entropy_loss, accuracy],
                                          feed_dict={
                                              samples: __batch_features,
                                              labels: __batch_labels
                                          })

            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f}."
            print(msg.format(__global_step, __accuracy, __loss))

        __accuracy, __global_step = sess.run([accuracy, global_step],
                                             feed_dict={
                                                 samples: x,
                                                 labels: y
                                             })

        print('Finetune accuracy: \t{}.'.format(__accuracy))


if __name__ == '__main__':
    x, y = load_dataset()

    # Train and save a model here.
    # train_model(x, y)

    load_model(x, y)
