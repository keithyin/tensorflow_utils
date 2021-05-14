
from __future__ import print_function
import tensorflow as tf
import tensorflow.nn as tfnn


def batch_norm_layer(x, is_train, decay=0.9, name_or_scope=None):
    """
    x: [b, emb_dim]
    """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="batch_norm_layer"):
        params_shape = [1, x.shape[-1]]
        beta = tf.get_variable("beta", params_shape, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable("gamma", params_shape, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        if is_train:
            mean, variance = tfnn.moments(x, axes=[0], keep_dims=True)
            moving_mean = tf.get_variable('moving_mean', shape=params_shape, dtype=tf.float32,
                                          initializer=tf.constant_initializer(
                                              0.0, tf.float32),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance', shape=params_shape, dtype=tf.float32,
                                              initializer=tf.constant_initializer(1.0, tf.float32),
                                              trainable=False)
            tf.add_to_collection(tf.GraphKeys.TRAIN_OP,
                                 tf.assign(moving_mean, decay * moving_mean + (1 - decay) * mean))
            tf.add_to_collection(tf.GraphKeys.TRAIN_OP,
                                 tf.assign(moving_variance, decay * moving_variance + (1 - decay) * variance))
        else:
            mean = tf.get_variable('moving_mean', shape=params_shape, dtype=tf.float32,
                                   initializer=tf.constant_initializer(
                                       0.0, tf.float32),
                                   trainable=False)
            variance = tf.get_variable('moving_variance', shape=params_shape, dtype=tf.float32,
                                       initializer=tf.constant_initializer(1.0, tf.float32),
                                       trainable=False)
        x = tfnn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)
    return x


if __name__ == '__main__':
    x = tf.random_uniform(shape=[2, 3], dtype=tf.float32)
    res = batch_norm_layer(x, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(res))
