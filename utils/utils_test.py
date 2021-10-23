
from __future__ import print_function
import tensorflow as tf
from unittest import TestCase
from utils import down_sampling_examples


class Test(TestCase):
    def test_eval_examples_down_sampling(self):
        feature_dict = {
            "age": tf.constant([1, 2, 3, 4], dtype=tf.int64),
            "height": tf.constant([1, 2, 3, 4], dtype=tf.int64),
            "coupon_info": tf.constant([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=tf.int64)
        }

        feature_dict = down_sampling_examples(feature_dict, sampling_rate=0.5)
        print(feature_dict)

        uniform = tf.random_uniform(shape=[10], minval=0, maxval=10, dtype=tf.int32)
        with tf.Session() as sess:
            print(sess.run(feature_dict))
            print(sess.run(uniform))
