import tensorflow as tf

'''Building the CIFAR-10 AlexNet

Summary of available functions:



'''

NUM_IMAGE_CHANNEL = 3
NUM_IMAGE_WIDTH = 32
NUM_IMAGE_HEIGHT = 32


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
                                 initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                                 dtype=tf.float32)
        bias = tf.get_variable('b', kernel_shape[3], initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(input, kernel, stride, 'SAME', data_format=data_format)
        pre_activation = tf.nn.bias_add(conv, bias, data_format=data_format)
        conv = tf.nn.relu(pre_activation, name=scope.name)

        tf.summary.histogram('Convolution_layers/{}_{}'.format(name, 'activation'), conv)
        tf.summary.scalar('Convolution_layers/{}_{}'.format(name, 'sparsity'), tf.nn.zero_fraction(conv))

    return conv

def fc_layer(input, size, name='fc'):
    '''Full Connected Layer in TensorFlow.

    :param input: A 2-D tensor of shape
        `[-1, NUM_INPUT]`.
    :param size: A list of `ints`.
        A 2-D tensor of shape
        `[NUM_INPUT, NUM_OUTPUT]`.
    :param name: An optional `string` for the name of this operation.
    :return:
        A TensorFlow operation of Full Connected Layer.
    '''
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('W', shape=size,
                                  initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32),
                                  dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

        biases = tf.get_variable('b', size[1], initializer=tf.constant_initializer(0.1))
        fc = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)
        tf.summary.histogram('Fully_connected_layers/{}_{}'.format(name, 'activation'), fc)
        tf.summary.scalar('Fully_connected_layers/{}_{}'.format(name, 'sparsity'), tf.nn.zero_fraction(fc))

    return fc


def inference(raw):
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

    conv1 = conv_layer(x, [5, 5, 3, 48], [1, 1, 1, 1],
                       data_format=data_format, name='conv1')

    norm1 = tf.nn.lrn(conv1, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    if data_format == 'NCHW':
        pool1 = tf.nn.max_pool(norm1, [1, 1, 3, 3], [1, 1, 2, 2], padding='SAME', data_format=data_format, name='pool1')
    else:
        pool1 = tf.nn.max_pool(norm1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format, name='pool1')

    conv2 = conv_layer(pool1, [5, 5, 48, 128], [1, 1, 1, 1],
                       data_format=data_format, name='conv2')

    norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    if data_format == 'NCHW':
        pool2 = tf.nn.max_pool(norm2, [1, 1, 3, 3], [1, 1, 2, 2], padding='SAME', data_format=data_format, name='pool2')
    else:
        pool2 = tf.nn.max_pool(norm2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format, name='pool2')

    conv3 = conv_layer(pool2, [3, 3, 128, 192], [1, 1, 1, 1],
                       data_format=data_format, name='conv3')

    conv4 = conv_layer(conv3, [3, 3, 192, 192], [1, 1, 1, 1],
                       data_format=data_format, name='conv4')

    conv5 = conv_layer(conv4, [3, 3, 192, 128], [1, 1, 1, 1],
                       data_format=data_format, name='conv5')

    norm3 = tf.nn.lrn(conv5, depth_radius=4, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    if data_format == 'NCHW':
        pool3 = tf.nn.max_pool(norm3, [1, 1, 3, 3], [1, 1, 2, 2], padding='SAME', data_format=data_format, name='pool3')
    else:
        pool3 = tf.nn.max_pool(norm3, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', data_format=data_format, name='pool3')





    return None
