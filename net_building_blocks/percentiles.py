from __future__ import print_function

import tensorflow as tf
import tensorflow.distributions as tfd


def gaussian_percentiles(loc, scale, percentiles):
    """
    Args:
        loc: mean, [batch_size, 1]
        scale:  std var, [batch_size, 1] or 1
        percentiles: int value
    Returns: tensor [batch_size, percentiles]
    """

    if loc.shape[-1] != 1:
        loc = tf.expand_dims(loc, -1)
    percentiles = [0+1.0/(percentiles + 1) * p for p in range(1, percentiles+1)]
    percentiles = tf.constant(percentiles, dtype=tf.float32)
    normal = tfd.Normal(loc=loc, scale=scale)
    return normal.quantile(percentiles)


if __name__ == '__main__':
    gaussian_loc = tf.constant([[0, 0.1]], dtype=tf.float32)
    gaussian_scale = tf.constant(0.2, dtype=tf.float32)
    res = gaussian_percentiles(gaussian_loc, 0.3, 5)
    with tf.Session() as sess:
        print(sess.run(res))
