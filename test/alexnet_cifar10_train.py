import tensorflow as tf

import alexnet_cifar10.input as input

CIFAR10_DIR = '/Users/Zhang/Research/Deep Learning Dataset/CIFAR'


def train():
    with tf.Session() as session:
        _training_images, _training_labels = input.create_batch_dataset(session,
                                                                        path=CIFAR10_DIR)

        _global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

    return None


if __name__ == '__main__':
    train()

    print 0
