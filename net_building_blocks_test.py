from __future__ import print_function

import tensorflow as tf
import unittest
from metis.tensorflow_utils.dag_core.net_building_blocks import attention


class Test(unittest.TestCase):
    def test_attention(self):
        q = tf.random_uniform(shape=[1, 2, 3])
        k = tf.random_uniform(shape=[1, 2, 3])
        v = tf.random_uniform(shape=[1, 2, 3])
        mask = tf.constant(value=[[1.0, 0.0]], dtype=tf.float32)
        res, attn_weights = attention.attention(
            q, k, v, v_mask=mask, attn_units=4, num_heads=2, ffn_filter_size=2, q_mask=mask, name_or_scope=None)
        print(res)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run([res, attn_weights]))
