from __future__ import print_function

import tensorflow as tf

a = tf.constant([0., 0.], dtype=tf.float32)

indicator = tf.greater(a, 0.5)

values = tf.constant([[1., 1.], [2., 2.]])

selected_vals = tf.boolean_mask(values, indicator)

selected_vals_mean = tf.reduce_mean(selected_vals, axis=0)

if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run([selected_vals, selected_vals_mean]))
