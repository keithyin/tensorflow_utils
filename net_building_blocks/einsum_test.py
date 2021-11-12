
from __future__ import print_function
from unittest import TestCase
import tensorflow as tf
from .mlp import n_experts
from .mtl import mmoe, mmoe_v2, mmoe_v3
import time
import numpy as np


class Test(TestCase):

    def test_einsum(self):
        """
        爱因斯坦标记法的核心就是，如果 同一个字母出现在两个操作数上，那么会对该字母累加求和
        这里是在测试，如果重复的字母出现在了输出上。那么仅仅是对应位置相乘，并没有求和
        Returns:

        """
        a = tf.constant([1, 2, 3], dtype=tf.float32)
        a = tf.reshape(a, shape=[1, 1, 3])
        b = tf.constant([9, 10, 11], dtype=tf.float32)
        b = tf.reshape(b, shape=[1, 1, 3])
        c = tf.einsum("nkg,khg->nhg", a, b)
        with tf.Session() as sess:
            print(sess.run(c))

    def test_einsum(self):
        a = tf.constant([1, 1, 1], dtype=tf.float32)
        a = tf.reshape(a, shape=[1, 1, 3])
        b = tf.constant([9, 10, 11], dtype=tf.float32)
        b = tf.reshape(b, shape=[1, 1, 3])
        c = tf.einsum("nkg,khg->nhg", a, b)
        with tf.Session() as sess:
            print(sess.run(c))

    def test_n_experts(self):
        x = tf.constant([[1, 2]], dtype=tf.float32)
        res = n_experts(x, hidden_sizes=[3, 4, 5], num_experts=3, activation=tf.nn.relu, use_bias=True)
        print(res)

    def test_mmoe(self):
        tf.reset_default_graph()
        x = tf.constant([[1, 2]], dtype=tf.float32)
        x = mmoe(x, num_experts=3, num_tasks=2, expert_hidden_sizes=[2, 2, 2], task_specific_hidden_sizes=[2, 1])
        print(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            all_times = []
            for i in range(50):
                begin_t = time.time()
                for _ in range(50000):
                    sess.run(x)
                all_times.append(time.time() - begin_t)
            print("v1_time: {}".format(np.array(all_times).mean()))

    def test_mmoe_v2(self):
        tf.reset_default_graph()
        x = tf.constant([[1, 2]], dtype=tf.float32)
        x = mmoe_v2(x, num_experts=3, num_tasks=2, expert_hidden_sizes=[2, 2, 2], task_specific_hidden_sizes=[2, 1])
        print(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            all_times = []
            for i in range(100):
                begin_t = time.time()
                for _ in range(50000):
                    sess.run(x)
                all_times.append(time.time() - begin_t)
            print("v2_time: {}".format(np.array(all_times).mean()))

    def test_mmoe_v3(self):
        tf.reset_default_graph()
        x = tf.constant([[1, 2]], dtype=tf.float32)
        x = mmoe_v3(x, num_experts=3, num_tasks=2, expert_hidden_sizes=[2, 2, 2], task_specific_hidden_sizes=[2, 1])
        print(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            all_times = []
            for i in range(100):
                begin_t = time.time()
                for _ in range(50000):
                    sess.run(x)
                all_times.append(time.time() - begin_t)
            print("v3_time: {}".format(np.array(all_times).mean()))
