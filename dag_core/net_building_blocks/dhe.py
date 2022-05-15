from __future__ import print_function

import math
from tqdm import tqdm
import numpy as np
import tensorflow as tf


def primes_in_range(low, high):
    def _is_prime(v):
        if v <= 3:
            return v > 1
        sqrt = int(math.sqrt(v))
        i = 2
        while i <= sqrt:
            if v % i == 0:
                return False
            i += 1
        return True

    return [v for v in tqdm(range(low + 1, high), desc="prime generator") if _is_prime(v)]


def dhe(x, k=1024, m=1e6):
    """

    Args:
        x: tensor, [batch, 1]
        k:
        m:

    Returns:

    """
    np.random.seed(10)
    a = np.random.randint(0, high=int(10 * m), size=k)
    b = np.random.randint(1, high=int(10 * m), size=k)
    primes = primes_in_range(int(m), int(5 * m))
    p = np.array(primes)[np.random.randint(0, len(primes), size=k)]
    # [7685385 8100989 5242852 ... 6036356 4850432 7258590] [9196472  594920 1018547 ... 7235258 8984625 7008918] [2834759 9429691 5537101 ... 4366669 9261647 5737213]
    print(a, b, p)
    if x.shape[-1] != 1:
        x = tf.expand_dims(x, axis=-1)
    a = tf.reshape(tf.constant(a), shape=[1, k])
    b = tf.reshape(tf.constant(b), shape=[1, k])
    p = tf.reshape(tf.constant(p), shape=[1, k])
    if len(x.shape) == 2:
        x = tf.tile(x, multiples=[1, k])
    elif len(x.shape) == 3:
        x = tf.tile(x, multiples=[1, 1, k])
        a = tf.expand_dims(a, axis=0)
        b = tf.expand_dims(b, axis=0)
        p = tf.expand_dims(p, axis=0)
    else:
        raise ValueError("not supported tensor shape")
    encod = tf.mod(tf.mod((a * x + b), p), int(m))
    encod = (tf.cast(encod, dtype=tf.float32) / float(m - 1) - 0.5) * 2.0
    return encod


if __name__ == '__main__':
    x = tf.constant([[1], [2]], dtype=tf.int64)
    res = dhe(x, m=10)
    print(res)

    with tf.Session() as sess:
        print(sess.run(res))
