import tensorflow as tf


class Inception(object):
    ################################################
    # Based on https://arxiv.org/pdf/1409.4842.pdf #
    ################################################

    def __init__(self, S, scope="inception"):
        self.S     = S
        self.scope = scope
    
    def __call__(self, x):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("1x1"):
                x1x1 = tf.layers.conv2d(x, self.S, 1, padding="same")
                x1x1 = tf.layers.batch_normalization(x1x1)
                x1x1 = tf.nn.relu(x1x1)

            with tf.variable_scope("1x1x3x3"):
                x1x1x3x3 = tf.layers.conv2d(x, self.S, 1, padding="same")
                x1x1x3x3 = tf.layers.batch_normalization(x1x1x3x3)
                x1x1x3x3 = tf.nn.relu(x1x1x3x3)

                x1x1x3x3 = tf.layers.conv2d(x1x1x3x3, self.S, 3, padding="same")
                x1x1x3x3 = tf.layers.batch_normalization(x1x1x3x3)
                x1x1x3x3 = tf.nn.relu(x1x1x3x3)

            with tf.variable_scope("1x1x5x5"):

                x1x1x5x5 = tf.layers.conv2d(x, self.S, 1, padding="same")
                x1x1x5x5 = tf.layers.batch_normalization(x1x1x5x5)
                x1x1x5x5 = tf.nn.relu(x1x1x5x5)

                x1x1x5x5 = tf.layers.conv2d(x1x1x5x5, self.S, 5, padding="same")
                x1x1x5x5 = tf.layers.batch_normalization(x1x1x5x5)
                x1x1x5x5 = tf.nn.relu(x1x1x5x5)

            with tf.variable_scope("MP1x1"):
                MP1x1 = tf.layers.max_pooling2d(x, 3, 1, padding="same")
                MP1x1 = tf.layers.conv2d(MP1x1, self.S, 1)
                MP1x1 = tf.layers.batch_normalization(MP1x1)
                MP1x1 = tf.nn.relu(MP1x1)

            return tf.concat(axis=3, values=[x1x1, x1x1x3x3, x1x1x5x5, MP1x1])

