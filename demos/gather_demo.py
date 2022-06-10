from __future__ import print_function

import tensorflow as tf

if __name__ == '__main__':

    a = tf.constant([[[1, 2, 3],
                      [4, 5, 6]
                      ],
                     [[1, 2, 3],
                      [4, 5, 6]
                      ]
                     ], dtype=tf.float32)

    a1 = tf.gather(a, indices=0, axis=1)
    a2 = tf.gather(a, indices=1, axis=1)
    print(a1.shape)

    print(a.shape)
    with tf.Session() as sess:
        print(sess.run(a1))
