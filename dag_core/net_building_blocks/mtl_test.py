
from __future__ import print_function
from unittest import TestCase
import tensorflow as tf
from .mtl import cgc, ple


class Test(TestCase):
    def test_cgc(self):
        num_tasks = 2
        num_experts = 4
        expert_hidden_sizes = [3]

        x = tf.random.normal(shape=[2, 3], dtype=tf.float32)

        res = cgc([x] * (num_tasks + 1), num_experts, num_tasks, expert_hidden_sizes)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print(sess.run(res))

    def test_ple(self):
        num_tasks = 2
        num_experts = 4
        expert_hidden_sizes = [3]
        task_specific_sizes = [3, 1]

        x = tf.random.normal(shape=[2, 3], dtype=tf.float32)

        res = ple(x, num_experts, num_tasks, expert_hidden_sizes, task_specific_sizes, num_cgc_layers=2)
        print(res)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print(sess.run(res))
