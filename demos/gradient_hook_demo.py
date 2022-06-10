from __future__ import print_function

import tensorflow as tf


# https://stackoverflow.com/questions/43839431/tensorflow-how-to-replace-or-modify-gradient
# https://stackoverflow.com/questions/54047604/how-to-assign-custom-gradient-to-tensorflow-op-with-multiple-inputs

@tf.custom_gradient
def clip_grad_layer(x, clip_min, clip_max):
    def grad(dy):
        return [tf.clip_by_value(dy, clip_value_min=clip_min, clip_value_max=clip_max),
                tf.constant(0.), tf.constant(0.)]

    return tf.identity(x), grad


a = tf.constant(2., dtype=tf.float32)
b = clip_grad_layer(a, clip_min=-0.1, clip_max=0.1)
# b = tf.identity(a)
c = 10 * b

da = tf.gradients(b, xs=[a])
if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run(da))
