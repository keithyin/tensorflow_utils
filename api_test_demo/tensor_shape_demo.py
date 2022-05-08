from __future__ import print_function

import tensorflow as tf

a = tf.ones(shape=[3, 2, 4], dtype=tf.float32)
b = a.shape[1].value
print(b, type(b))

c = a / b

if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run(c))