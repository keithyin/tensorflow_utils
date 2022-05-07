from __future__ import print_function
import tensorflow as tf
import tensorflow.nn as tfnn


def dice(x, is_train, alpha, decay=0.9, name_or_scope=None):
    """
    x: [b, emb_dim]
    """
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="DiceLayer"):
        params_shape = [1, x.shape[-1]]
        assert len(x.shape) == 2, "only support 2-Dim"

        if is_train:
            mean, variance = tfnn.moments(x, axes=[0], keep_dims=True)
            moving_mean = tf.get_variable(
                'moving_mean', shape=params_shape, dtype=tf.float32,
                initializer=tf.constant_initializer(
                    0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', shape=params_shape, dtype=tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            tf.add_to_collection(
                tf.GraphKeys.TRAIN_OP,
                tf.assign(moving_mean, decay * moving_mean + (1 - decay) * mean))
            tf.add_to_collection(
                tf.GraphKeys.TRAIN_OP,
                tf.assign(moving_variance, decay * moving_variance + (1 - decay) * variance))
        else:
            mean = tf.get_variable(
                'moving_mean', shape=params_shape, dtype=tf.float32,
                initializer=tf.constant_initializer(
                    0.0, tf.float32),
                trainable=False)
            variance = tf.get_variable(
                'moving_variance', shape=params_shape, dtype=tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
        ps = tf.div(1.0, 1.0 + (tf.exp(-(x - mean) / tf.sqrt(variance + 1e-6))))
        x = ps * x + (1 - ps) * alpha * x
    return x


if __name__ == '__main__':
    x = tf.random_uniform(shape=[2, 3], dtype=tf.float32)
    res = dice(x, True, 0.1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(res))
