from __future__ import print_function

import tensorflow as tf


def sigmoid_loss(label, logit):
    return - (label * tf.log(tf.sigmoid(logit)) + (1-label) * tf.log(1-tf.sigmoid(logit)))


label = tf.constant(value=[0.2], dtype=tf.float32)

logit = tf.constant(value=[-1.0], dtype=tf.float32)

loss = tf.losses.sigmoid_cross_entropy(label, logit)

loss2 = sigmoid_loss(label, logit)

tf.Variable().load()

if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run([loss, loss2]))
