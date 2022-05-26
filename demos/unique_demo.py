from __future__ import print_function
import tensorflow as tf

a = tf.constant([[1, 2, 3, 4, 1, 2], [1, 2, 3, 4, 1, 2]], dtype=tf.int64)
b = tf.unique(a)

if __name__ == '__main__':

    with tf.Session() as sess:
        print(sess.run(b))
