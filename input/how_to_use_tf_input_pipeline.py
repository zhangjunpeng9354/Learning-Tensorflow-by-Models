import os

import tensorflow as tf
import numpy as np

'''Explaining how to custom input for tensorflow models.


'''


def input_pipe_fn(session):
    dataset = tf.contrib.data.Dataset.from_tensor_slices(
        tf.reshape(np.arange(200), [-1, 2])
    )

    dataset = dataset.batch(4)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    session.run(iterator.initializer)

    return next_element


def training_fn(batch):
    return batch + 1


def evaluation_fn(batch):
    return batch - 1


if __name__ == '__main__':
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        _batch = input_pipe_fn(session)

        train_op = training_fn(_batch)
        evaluation_op = evaluation_fn(_batch)

        _batch_ouput, _train_ouput, _evaluation_output = session.run([_batch, train_op, evaluation_op])

        print(_batch_ouput)
        print('+++' * 40)
        print(_train_ouput)
        print('+++' * 40)
        print(_evaluation_output)

        print('---' * 40)

        _batch_ouput, _train_ouput = session.run([_batch, train_op])

        print(_batch_ouput)
        print('+++' * 40)
        print(_train_ouput)
        print('+++' * 40)

        _batch_ouput, _evaluation_output = session.run([_batch, evaluation_op])

        print(_batch_ouput)
        print('+++' * 40)
        print(_evaluation_output)
