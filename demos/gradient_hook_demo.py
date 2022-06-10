from __future__ import print_function

import tensorflow as tf


# https://stackoverflow.com/questions/43839431/tensorflow-how-to-replace-or-modify-gradient

@tf.custom_gradient
def clip_grad_layer(x):
    def grad(dy):
        return tf.clip_by_value(dy, -0.1, 0.1)

    return tf.identity(x), grad


a = tf.constant(2., dtype=tf.float32)
# b = clip_grad_layer(a)
b = tf.identity(a)
c = 10 * b

da = tf.gradients(b, xs=[a])
if __name__ == '__main__':

    with tf.Session() as sess:
        print(sess.run(da))
