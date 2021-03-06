# coding=utf-8
from __future__ import print_function
from unittest import TestCase
import tensorflow as tf
import numpy as np
from .losses import exponential_nll_loss, inverse_gaussian_nll_loss


class Test(TestCase):
    def test_exponential_nll_loss(self):
        tf.reset_default_graph()
        data = np.random.exponential(0.1, 100000)

        print(data)
        inp = tf.constant(data, dtype=tf.float32, shape=data.shape)
        lam = tf.exp(tf.get_variable("lambda", shape=[], dtype=tf.float32))
        loss = tf.reduce_mean(exponential_nll_loss(lam, inp))
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                print(i, sess.run([train_op, loss, lam]))

        print(1 / np.mean(data))

    def test_inverse_gaussian_nll_loss(self):

        tf.reset_default_graph()
        data = np.random.wald(mean=10, scale=2, size=10000)

        print(data)
        inp = tf.constant(data, dtype=tf.float32, shape=data.shape)
        lam = tf.exp(tf.get_variable("lambda", shape=[], dtype=tf.float32))
        mu = tf.exp(tf.get_variable("mu", shape=[], dtype=tf.float32))
        # TODO: mu能学出来，lambda学不出来，不知道为啥。。。。。
        loss = tf.reduce_mean(inverse_gaussian_nll_loss(lam, mu, inp))
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                print(i, sess.run([train_op, loss, lam, mu]))

        print(np.mean(data))
