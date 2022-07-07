from __future__ import print_function

import functools

import tensorflow as tf


def text_cnn(inputs, name_or_scope, reuse):
    """

    Args:
        inputs: [b, T, dim]
        name_or_scope:
        reuse: boolean

    Returns:

    """

    # [b, dim]
    with tf.variable_scope(name_or_scope=name_or_scope, default_name="TextCnn", reuse=reuse):
        x_1 = tf.reduce_max(
            tf.layers.conv1d(inputs=inputs, filters=128, kernel_size=1, activation=tf.nn.relu), axis=1, keep_dims=False)
        x_2 = tf.reduce_max(
            tf.layers.conv1d(inputs=inputs, filters=256, kernel_size=2, activation=tf.nn.relu), axis=1, keep_dims=False)
        x_3 = tf.reduce_max(
            tf.layers.conv1d(inputs=inputs, filters=512, kernel_size=3, activation=tf.nn.relu), axis=1, keep_dims=False)

        x = tf.concat([x_1, x_2, x_3], axis=1)
    return x


if __name__ == '__main__':
    query = tf.random.uniform(shape=[2, 15, 8])
    brand = tf.random.uniform(shape=[2, 15, 8])

    query_emb = text_cnn(query, name_or_scope="TextCnn", reuse=tf.AUTO_REUSE)
    brand_emb = text_cnn(brand, name_or_scope="TextCnn", reuse=tf.AUTO_REUSE)

    brand_emb = tf.layers.dense(brand_emb, units=2, activation=functools.partial(tf.nn.l2_normalize, axis=-1))

    cop = tf.reduce_sum(brand_emb ** 2, axis=1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(brand_emb))
        print(sess.run(cop))