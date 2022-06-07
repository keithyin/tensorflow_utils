from __future__ import print_function
from unittest import TestCase
import tensorflow as tf
import ipm


class Test(TestCase):
    def test_mmd2_lin(self):
        x = tf.random.uniform(shape=[3, 4], dtype=tf.float32)
        t = tf.constant(value=[[1], [0], [1]], dtype=tf.int64)

        res = ipm.mmd2_lin(x, t, p=0.5)

        it = tf.where(t > 0)

        gathered_t_rep = tf.gather(x, it[:, 0])

        print(it)
        with tf.Session() as sess:
            print(sess.run([res, it, gathered_t_rep, x]))
