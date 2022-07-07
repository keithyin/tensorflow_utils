from __future__ import print_function

import tensorflow as tf

a = tf.constant("hello", dtype=tf.string)
if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run(a))