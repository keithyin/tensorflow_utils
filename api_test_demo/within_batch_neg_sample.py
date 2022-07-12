from __future__ import print_function

import tensorflow as tf

a = tf.nn.l2_normalize(tf.random.uniform(shape=[3, 2], dtype=tf.float32), axis=-1)
b = tf.nn.l2_normalize(tf.random.uniform(shape=[3, 2], dtype=tf.float32), axis=-1)
identity_matrix = tf.eye(num_rows=tf.shape(a)[0], dtype=tf.float32)
dist = 0.5 * tf.sqrt(2 * (1-tf.einsum("ai,bi->ab", a, b))) + identity_matrix
top_dist = -tf.math.top_k(-dist, k=1).values

loss = tf.reduce_mean(top_dist)
grads = tf.gradients(loss, xs=[dist])

if __name__ == '__main__':

    print(dist)
    with tf.Session() as sess:
        print(sess.run(top_dist))
        print(sess.run(grads))