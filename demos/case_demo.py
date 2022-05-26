# coding=utf-8
from __future__ import print_function

import tensorflow as tf


def get_inputs_weights(x, values, weights):
    """
    根据 values 和 weights 给 x 中的每个位置赋值
    Args:
        x: input, tensor
        values: int list
        weights: float list, 表示 values 对应位置数值的权重

    Returns: x 中每个位置的权重

    """
    result_weights = tf.ones_like(x, dtype=tf.float32)
    for v, w in zip(values, weights):
        result_weights = tf.where(tf.equal(x, tf.constant(v, dtype=tf.int64)),
                                  tf.ones_like(x, dtype=tf.float32) * w, result_weights)
    return result_weights


if __name__ == '__main__':
    x = tf.constant([1, 2, 3], dtype=tf.int64)
    vals = [1, 2, 3]
    ws = [0.1, 0.2, 0.3]

    res = get_inputs_weights(x, vals, ws)

    with tf.Session() as sess:
        print(sess.run(res))
