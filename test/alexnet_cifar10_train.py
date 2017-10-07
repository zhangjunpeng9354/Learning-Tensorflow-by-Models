import sys

sys.path.append('../')

from time import time

import tensorflow as tf

import alexnet_cifar10.input as input
import alexnet_cifar10.model as model

CIFAR10_DIR = '/Users/Zhang/Research/Deep Learning Dataset/CIFAR/cifar-10-batches-py'

BATCH_SIZE = 128
NUM_ITERATION = 20000


def train():
    with tf.Session() as session:



        _training_images, _training_labels = input.create_batch_dataset(session,
                                                                        BATCH_SIZE,
                                                                        path=CIFAR10_DIR)

        print(_training_labels.shape)

        _global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

        # Capture Tensorflow operations for required computation graphs.
        _logits_op = model.inference(_training_images)

        _loss_op = model.loss(_logits_op, _training_labels)

        _training_accuracy_op = model.evaluate(_logits_op, _training_labels, name='Training_Accuracy')

        _train_op = model.train(_loss_op, _global_step)

        session.run(tf.global_variables_initializer())

        for _ in range(NUM_ITERATION):
            start_time = time()
            __global_step, _ = session.run([_global_step, _training_accuracy_op])
            duration = time() - start_time

            if (__global_step % 10) == 0:
                __loss, __training_accuracy = session.run([_loss_op, _training_accuracy_op])
                _msg_training = 'Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)'
                print(_msg_training.format(__global_step, __training_accuracy, __loss, BATCH_SIZE / duration, duration))

    return None


if __name__ == '__main__':
    train()

    print 0
