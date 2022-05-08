from __future__ import print_function

import tensorflow as tf

if __name__ == '__main__':
    mask = tf.constant([[1, 1, 0], [1, 0, 0]], dtype=tf.float32)
    value = tf.ones(shape=[2, 3, 2])
    res = tf.einsum("nt,ntd->nd", mask, value)
    with tf.Session() as sess:
        print(sess.run(res))