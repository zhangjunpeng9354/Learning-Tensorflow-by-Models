from time import time

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


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


def AlexNet_model_fn():
    NUM_IMAGE_WIDTH = 32
    NUM_IMAGE_HEIGHT = 32
    NUM_IMAGE_CHANNEL = 3
    NUM_CLASS = 10

    with tf.variable_scope('input'):
        x_raw = tf.placeholder(dtype=tf.float32, shape=[None, NUM_IMAGE_WIDTH * NUM_IMAGE_HEIGHT * NUM_IMAGE_CHANNEL],
                               name='raw_images')
        x = tf.reshape(x_raw, shape=[-1, NUM_IMAGE_CHANNEL, NUM_IMAGE_WIDTH, NUM_IMAGE_HEIGHT], name='input_images')
        data_format = 'NCHW'

        y = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASS], name='input_onehot_labels')

        if tf.test.is_built_with_cuda() is not True:
            # When running on GPU, transpose the data from channels_last (NHWC) to
            # channels_first (NCHW) to improve performance.
            # See https://www.tensorflow.org/performance/performance_guide#data_formats
            data_format = 'NHWC'
            x = tf.transpose(x, [0, 2, 3, 1])

    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('W', shape=[5, 5, 3, 48],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                                 dtype=tf.float32)
        bias = tf.get_variable('b', [48], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME', data_format=data_format)
        pre_activation = tf.nn.bias_add(conv, bias, data_format=data_format)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    if data_format == 'NCHW':
        pool1 = tf.nn.max_pool(norm1, [1, 1, 3, 3], [1, 1, 2, 2], padding='SAME', data_format=data_format, name='pool1')
    else:
        pool1 = tf.nn.max_pool(norm1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format, name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('W', shape=[5, 5, 48, 128],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                                 dtype=tf.float32)
        bias = tf.get_variable('b', [128], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], 'SAME', data_format=data_format)
        pre_activation = tf.nn.bias_add(conv, bias, data_format=data_format)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    if data_format == 'NCHW':
        pool2 = tf.nn.max_pool(norm2, [1, 1, 3, 3], [1, 1, 2, 2], padding='SAME', data_format=data_format, name='pool2')
    else:
        pool2 = tf.nn.max_pool(norm2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format, name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = tf.get_variable('W', shape=[3, 3, 128, 192],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                                 dtype=tf.float32)
        bias = tf.get_variable('b', [192], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], 'SAME', data_format=data_format)
        pre_activation = tf.nn.bias_add(conv, bias, data_format=data_format)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    with tf.variable_scope('conv4') as scope:
        kernel = tf.get_variable('W', shape=[3, 3, 192, 192],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                                 dtype=tf.float32)
        bias = tf.get_variable('b', [192], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], 'SAME', data_format=data_format)
        pre_activation = tf.nn.bias_add(conv, bias, data_format=data_format)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)

    tf.summary.histogram('Convolution_layers/conv4', conv4)
    tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

    with tf.variable_scope('conv5') as scope:
        kernel = tf.get_variable('W', shape=[3, 3, 192, 128],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                                 dtype=tf.float32)
        bias = tf.get_variable('b', [128], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], 'SAME', data_format=data_format)
        pre_activation = tf.nn.bias_add(conv, bias, data_format=data_format)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)

    tf.summary.histogram('Convolution_layers/conv5', conv5)
    tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv5, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    if data_format == 'NCHW':
        pool3 = tf.nn.max_pool(norm3, [1, 1, 3, 3], [1, 1, 2, 2], padding='SAME', data_format=data_format, name='pool1')
    else:
        pool3 = tf.nn.max_pool(norm3, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format, name='pool1')

    with tf.variable_scope('fully_connected1') as scope:
        pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 128])

        weights = tf.get_variable('W', shape=[4 * 4 * 128, 384],
                                  initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32),
                                  dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

        biases = tf.get_variable('b', [384], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(pool3_flat, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully_connected_layers/fc1', fc1)
    tf.summary.scalar('Fully_connected_layers/fc1', tf.nn.zero_fraction(fc1))

    with tf.variable_scope('fully_connected2') as scope:
        weights = tf.get_variable('W', shape=[384, 192],
                                  initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32),
                                  dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

        biases = tf.get_variable('b', [192], initializer=tf.constant_initializer(0.))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully_connected_layers/fc2', fc2)
    tf.summary.scalar('Fully_connected_layers/fc2', tf.nn.zero_fraction(fc2))

    with tf.variable_scope('fully_connected3') as scope:
        weights = tf.get_variable('W', shape=[192, NUM_CLASS],
                                  initializer=tf.truncated_normal_initializer(stddev=1 / 192.0, dtype=tf.float32),
                                  dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

        biases = tf.get_variable('b', [NUM_CLASS], initializer=tf.constant_initializer(0.001))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)
    tf.summary.histogram('Fully_connected_layers/fc3', softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(softmax_linear, axis=1)

    return x_raw, y, softmax_linear, global_step, y_pred_cls


if __name__ == "__main__":
    x, y, output, global_step, y_pred_cls = AlexNet_model_fn()

    with tf.variable_scope('loss'):
        cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                                    logits=output),
                                            name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    optimizer = tf.train.RMSPropOptimizer(0.0001).minimize(loss, global_step=global_step)

    with tf.variable_scope('accuracy'):
        correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy/train', accuracy)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess = tf.Session()
    tf_train_writer = tf.summary.FileWriter('./tensorbroad/', sess.graph)

    sess.run(tf.global_variables_initializer())

    train_images, train_labels, test_images, test_labels = load_cifar10()

    NUM_ITERATION = 20000
    SIZE_BATCH = 128

    for iter in range(NUM_ITERATION):
        randidx = np.random.randint(len(train_images), size=SIZE_BATCH)
        batch_train_images = train_images[randidx]
        batch_train_labels = train_labels[randidx]

        start_time = time()
        _global_step, _ = sess.run([global_step, optimizer],
                                   feed_dict={x: batch_train_images,
                                              y: batch_train_labels})
        duration = time() - start_time

        if (_global_step % 10 == 0) or (iter == NUM_ITERATION - 1):
            _loss, batch_acc = sess.run([loss, accuracy], feed_dict={x: batch_train_images, y: batch_train_labels})
            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
            print(msg.format(_global_step, batch_acc, _loss, SIZE_BATCH / duration, duration))

        if (_global_step % 100 == 0) or (iter == NUM_ITERATION - 1):
            data_merged, global_step_iter = sess.run([merged, global_step],
                                                     feed_dict={x: batch_train_images,
                                                                y: batch_train_labels})

            predicted_class = sess.run(y_pred_cls, feed_dict={x: test_images, y: test_labels})

            correct = (np.argmax(test_labels, axis=1) == predicted_class)
            acc = correct.mean() * 100
            correct_numbers = correct.sum()
            print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_labels)))

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
            ])
            tf_train_writer.add_summary(data_merged, global_step_iter)
            tf_train_writer.add_summary(summary, global_step_iter)

            saver.save(sess, save_path='./tensorbroad/', global_step=global_step)
            print("Saved checkpoint.")

    sess.close()
