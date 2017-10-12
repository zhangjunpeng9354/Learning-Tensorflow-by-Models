from time import time
import math

import numpy as np
import tensorflow as tf

'''Building the CIFAR-10 VGG-Net

Summary of available functions:



'''

BATCH_SIZE = 64
NUM_ITERATION = 60000

NUM_CLASS = 10
NUM_IMAGE_CHANNEL = 3
NUM_IMAGE_WIDTH = 32
NUM_IMAGE_HEIGHT = 32

CACHE_DIR = '/home/ubuntu/notebook/tensorboard/vgg-cifar10'


# CACHE_DIR = '/Users/Zhang/Research/Programming/Learning-Tensorflow-by-Models'


def load_cifar10(path=None):
    if path is None:
        # DATASET_DIR = '/Users/Zhang/Research/Deep Learning Dataset/CIFAR/cifar-10-batches-py/'
        DATASET_DIR = '/home/ubuntu/datasets/cifar-10-batches-py/'
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

    return train_images.astype(np.float32), train_labels, test_images.astype(np.float32), test_labels


def conv_layer(input, kernel_shape, stride, data_format='NCHW', name='conv'):
    '''2-D Convolution Layer Operation in TensorFlow.

    :param input: A 4-D tensor. The dimension order is determined by the value of
        `data_format`.
    :param kernel_shape: A list of `ints`.
        A 4-D tensor of shape
        `[filter_height, filter_width, in_channels, out_channels]`.
    :param stride: A list of `ints`.
        1-D tensor of length 4. The stride of the sliding window for each
        dimension of `input`. The dimension order is determined by the value of
        `data_format`.
    :param data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `NCHW`.
        Specify the data format of the input and output data. With the
        default format "NCHW", the data is stored in the order of:
        [batch, channels, height, width].
    :param name: An optional `string` for the name of this operation.
    :return:
        A TensorFlow operation of 2-D Convolution Layer.
    '''
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('W', shape=kernel_shape,
                                 initializer=tf.truncated_normal_initializer(
                                     stddev=math.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])),
                                     dtype=tf.float32),
                                 dtype=tf.float32)
        # kernel = tf.get_variable('W', shape=kernel_shape,
        #                          initializer=tf.truncated_normal_initializer(
        #                              stddev=0.05,
        #                              dtype=tf.float32),
        #                          dtype=tf.float32)
        bias = tf.get_variable('b', kernel_shape[3], initializer=tf.constant_initializer(0.001))

        conv = tf.nn.conv2d(input, kernel, stride, 'SAME', data_format=data_format)
        pre_activation = tf.nn.bias_add(conv, bias, data_format=data_format)
        conv = tf.nn.relu(pre_activation, name=scope.name)

        tf.summary.histogram('Convolution_layers/{}_{}'.format(name, 'activation'), conv)
        tf.summary.scalar('Convolution_layers/{}_{}'.format(name, 'sparsity'), tf.nn.zero_fraction(conv))

    return conv


def fc_layer(input, size, name='fc', final=False):
    '''Full Connected Layer in TensorFlow.

    :param input: A 2-D tensor of shape
        `[-1, NUM_INPUT]`.
    :param size: A list of `ints`.
        A 2-D tensor of shape
        `[NUM_INPUT, NUM_OUTPUT]`.
    :param final: An optinal `bool`. Default is `False`, while `True` is for the final layer.
    :param name: An optional `string` for the name of this operation.
    :return:
        A TensorFlow operation of Full Connected Layer.
    '''
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('W', shape=size,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=math.sqrt(1.0 / (size[0]+size[1])),
                                      dtype=tf.float32),
                                  dtype=tf.float32)
        # weights = tf.get_variable('W', shape=size,
        #                           initializer=tf.truncated_normal_initializer(
        #                               stddev=0.01,
        #                               dtype=tf.float32),
        #                           dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.0005, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

        biases = tf.get_variable('b', size[1], initializer=tf.constant_initializer(0.0001))

        if final is True:
            fc = tf.add(tf.matmul(input, weights), biases, name=scope.name)
        else:
            fc = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)
            tf.summary.histogram('Fully_connected_layers/{}_{}'.format(name, 'activation'), fc)
            tf.summary.scalar('Fully_connected_layers/{}_{}'.format(name, 'sparsity'), tf.nn.zero_fraction(fc))

    return fc


def inference(raw, keep_prob):
    '''

    :param raw:
    :return:
    '''
    with tf.variable_scope('input') as scope:
        # When running on GPU, transpose the data from channels_last (NHWC) to
        # channels_first (NCHW) to improve performance.
        # See https://www.tensorflow.org/performance/performance_guide#data_formats
        x = tf.reshape(raw, shape=[-1, NUM_IMAGE_CHANNEL, NUM_IMAGE_WIDTH, NUM_IMAGE_HEIGHT], name='input_images')
        data_format = 'NCHW'

        if tf.test.is_built_with_cuda() is not True:
            data_format = 'NHWC'
            x = tf.transpose(x, [0, 2, 3, 1])

    # convolution group 1, output - [16, 16, 64]
    conv1 = conv_layer(x, [3, 3, 3, 64], [1, 1, 1, 1],
                       data_format=data_format, name='conv1')
    norm1 = tf.nn.lrn(conv1, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    if data_format == 'NCHW':
        pool1 = tf.nn.max_pool(norm1, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format=data_format, name='pool1')
    else:
        pool1 = tf.nn.max_pool(norm1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format, name='pool1')

    # convolution group 2, output - [8, 8, 128]
    conv2 = conv_layer(pool1, [3, 3, 64, 128], [1, 1, 1, 1],
                       data_format=data_format, name='conv2')
    norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    if data_format == 'NCHW':
        pool2 = tf.nn.max_pool(norm2, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format=data_format, name='pool2')
    else:
        pool2 = tf.nn.max_pool(norm2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format, name='pool2')

    # convolution group 3, output - [4, 4, 256]
    conv3_1 = conv_layer(pool2, [3, 3, 128, 256], [1, 1, 1, 1],
                         data_format=data_format, name='conv3_1')
    norm3_1 = tf.nn.lrn(conv3_1, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm3_1')
    conv3_2 = conv_layer(norm3_1, [3, 3, 256, 256], [1, 1, 1, 1],
                         data_format=data_format, name='conv3_2')
    norm3_2 = tf.nn.lrn(conv3_2, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm3_2')
    if data_format == 'NCHW':
        pool3 = tf.nn.max_pool(norm3_2, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format=data_format,
                               name='pool3')
    else:
        pool3 = tf.nn.max_pool(norm3_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format,
                               name='pool3')

    # convolution group 4, output - [2, 2, 512]
    conv4_1 = conv_layer(pool3, [3, 3, 256, 512], [1, 1, 1, 1],
                         data_format=data_format, name='conv4_1')
    norm4_1 = tf.nn.lrn(conv4_1, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm4_1')
    conv4_2 = conv_layer(norm4_1, [3, 3, 512, 512], [1, 1, 1, 1],
                         data_format=data_format, name='conv4_2')
    norm4_2 = tf.nn.lrn(conv4_2, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm4_2')
    if data_format == 'NCHW':
        pool4 = tf.nn.max_pool(norm4_2, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format=data_format,
                               name='pool4')
    else:
        pool4 = tf.nn.max_pool(norm4_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format,
                               name='pool4')

    # convolution group 5, output - [1, 1, 512]
    conv5_1 = conv_layer(pool4, [3, 3, 512, 512], [1, 1, 1, 1],
                         data_format=data_format, name='conv5_1')
    norm5_1 = tf.nn.lrn(conv5_1, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm5_1')
    conv5_2 = conv_layer(norm5_1, [3, 3, 512, 512], [1, 1, 1, 1],
                         data_format=data_format, name='conv5_2')
    norm5_1 = tf.nn.lrn(conv5_2, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm5_2')
    if data_format == 'NCHW':
        pool5 = tf.nn.max_pool(norm5_1, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format=data_format,
                               name='pool5')
    else:
        pool5 = tf.nn.max_pool(norm5_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format,
                               name='pool5')

    pool5_flat = tf.reshape(pool5, [-1, 1 * 1 * 512], name='flatten')

    fc1 = fc_layer(pool5_flat, [1 * 1 * 512, 128], name='fc1', final=False)

    droput1 = tf.nn.dropout(fc1, keep_prob)

    fc2 = fc_layer(droput1, [128, 64], name='fc2', final=False)

    droput2 = tf.nn.dropout(fc2, keep_prob)

    softmax_linear = fc_layer(droput2, [64, NUM_CLASS], name='fc3', final=True)

    return softmax_linear


def loss(logits, labels):
    '''

    :param logits:
    :param labels:
    :return:
    '''
    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss


def train(total_loss, global_step):
    # Decay the learning rate exponentially based on the number of steps. best
    lr = tf.train.exponential_decay(0.005,
                                    global_step,
                                    500,
                                    0.9,
                                    staircase=True)

    # lr = tf.train.exponential_decay(0.001,
    #                                 global_step,
    #                                 2500,
    #                                 0.316,
    #                                 staircase=True) not good

    # lr = 0.0005
    tf.summary.scalar('learning_rate/lr', lr)

    optimizer = tf.train.RMSPropOptimizer(lr)
    grads = optimizer.compute_gradients(total_loss)

    appply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
        if var is not None:
            tf.summary.histogram(var.op.name, var)

    return appply_gradient_op


def evaluate(logits, labels, name='Train'):
    y_pred_cls = tf.argmax(logits, axis=1)
    correct_prediction = tf.equal(y_pred_cls, tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('Accuracy/{}'.format(name), accuracy)

    return accuracy


if __name__ == '__main__':
    # load data
    train_images, train_labels, test_images, test_labels = load_cifar10()
    train_images = (train_images - 128) / 255.0
    test_images = (test_images - 128) / 255.0

    # build variables for training procedure.
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # build train operation and variables.
    train_x = tf.placeholder(tf.float32, shape=[None, NUM_IMAGE_WIDTH * NUM_IMAGE_HEIGHT * NUM_IMAGE_CHANNEL],
                             name='train_images')
    train_y = tf.placeholder(tf.float32, shape=[None, NUM_CLASS], name='train_label')

    train_logits = inference(train_x, keep_prob)
    loss_op = loss(train_logits, train_y)
    train_op = train(loss_op, global_step)

    accuacy_op = evaluate(train_logits, train_y, name='Train')

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        tf_train_writer = tf.summary.FileWriter(CACHE_DIR, session.graph)

        for iter in range(NUM_ITERATION):

            randidx = np.random.randint(len(train_images), size=BATCH_SIZE)
            batch_train_images = train_images[randidx]
            batch_train_labels = train_labels[randidx]

            start_time = time()
            _global_step, _ = session.run([global_step, train_op],
                                          feed_dict={train_x: batch_train_images,
                                                     train_y: batch_train_labels,
                                                     keep_prob: 0.5})
            duration = time() - start_time

            if (iter + 1) % 10 == 0:
                _loss, _train_accuracy = session.run([loss_op, accuacy_op],
                                                     feed_dict={train_x: batch_train_images,
                                                                train_y: batch_train_labels,
                                                                keep_prob: 1.0})
                msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
                print(msg.format(_global_step, _train_accuracy, _loss, BATCH_SIZE / duration, duration))

            if (iter + 1) % 100 == 0:
                data_merged, global_step_iter = session.run([merged, global_step],
                                                            feed_dict={train_x: batch_train_images,
                                                                       train_y: batch_train_labels,
                                                                       keep_prob: 1.0}
                                                            )

                _eval_accuracy = session.run(accuacy_op,
                                             feed_dict={train_x: test_images,
                                                        train_y: test_labels,
                                                        keep_prob: 1.0})

                print("Accuracy on Test-Set: {0:.2f}%".format(_eval_accuracy * 100.0))

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="Accuracy/Test", simple_value=_eval_accuracy),
                ])
                tf_train_writer.add_summary(data_merged, global_step_iter)
                tf_train_writer.add_summary(summary, global_step_iter)

                saver.save(session, save_path=CACHE_DIR, global_step=global_step)
                print("Saved checkpoint.")
