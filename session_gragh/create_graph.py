import tensorflow as tf


def create_graph():
    # Create a empty tf.Graph and print it.
    g = tf.Graph()
    print('A Graph g is created. \t {}'.format(g))

    # Does g represent a function or not? No as long as it is just created.
    print('Does Graph g represent a function? \t{}'.format(g.building_function))

    # Is g the default graph? No.
    # By using the as_default() g override the default graph.
    print('Is Graph g is the default graph? \t{}'.format(g is tf.get_default_graph()))
    print('Information on the default graph:\t {}'.format(tf.get_default_graph()))
    print('Information on the g:\t {}'.format(g))

    # Try to override the default graph.
    print('-' * 35)
    print('Try to override the default graph.')
    with g.as_default():
        print('Is Graph g is the default graph? \t{}'.format(g is tf.get_default_graph()))
        print('Information on the default graph:\t {}'.format(tf.get_default_graph()))
        print('Information on the g:\t {}'.format(g))

        a_constant = tf.constant(0.001, dtype=tf.float32)
        print(a_constant.graph)
        print('Does Graph g represent a function? \t{}'.format(g.building_function))

        x = tf.placeholder(dtype=tf.float32)
        y = x + 1
        print(y.graph)
        print(y)
        print('Does Graph g represent a function? \t{}'.format(g.building_function))

        print('Is y an element in g? \t{}'.format(g.as_graph_element(y, allow_tensor=True, allow_operation=False)))

        key_list = g.get_all_collection_keys()
        for key in key_list:
            print('\t{}'.format(key))

    print('Information on the default graph:\t {}'.format(tf.get_default_graph()))

    print('=' * 35)


if __name__ == '__main__':
    create_graph()
