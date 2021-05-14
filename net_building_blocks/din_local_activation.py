from __future__ import print_function

import tensorflow as tf
import numpy as np


def din_local_activation_block(user_behaviors, mask, query):
    """
    :param user_behaviors: [b, T, sub_field, emb_size] or [b, T, emb_size]
    :param mask: [b, T, sub_field] or [b, T]
    :param query: [b, num_sub_field, emb_size] or [b, emb_size]
    """
    with tf.variable_scope(name_or_scope=None, default_name="din_local_activation_block"):
        batch_size = tf.shape(user_behaviors)[0]
        time_step = tf.shape(user_behaviors)[1]

        user_behaviors = tf.reshape(user_behaviors, shape=[batch_size, time_step, np.prod(user_behaviors.shape[2:])])
        if len(mask.shape) == 3:
            mask = tf.reduce_mean(mask, axis=2, keepdims=True)
        elif len(mask.shape) == 2:
            mask = tf.expand_dims(mask, axis=2)

        query = tf.reshape(query, shape=[batch_size, np.prod(query.shape[1:])])

        # user_behaviors: [b, T, hidden_size], mask: [b, T, 1], query: [b, hidden_size]
        assert len(user_behaviors.shape) == 3
        assert len(mask.shape) == 3
        assert len(query.shape) == 2

        assert user_behaviors.shape[2] == query.shape[1], """
            user_behavior and query must share the same hidden_size, 
            but got user_behavior:{}, query:{}""".format(user_behaviors, query)

        # [b, 1, hidden_size]
        query = tf.expand_dims(query, axis=1)
        query = tf.tile(query, multiples=[1, time_step, 1])

        # [b, T, hidden_size + hidden_size + hidden_size]
        state = tf.concat([query, user_behaviors, query * user_behaviors], axis=2)

        x = tf.layers.dense(state, 128, activation=tf.nn.relu)

        # [b, T, 1]
        weights = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, use_bias=False)
        weights = tf.multiply(weights, mask, name="weights")

        activated_behavior = tf.reduce_sum(weights * user_behaviors, axis=1, keepdims=False)
    return activated_behavior


if __name__ == '__main__':
    user_beh = tf.random_uniform(shape=[2, 2, 1, 6], dtype=tf.float32)

    query = tf.random_uniform(shape=[2, 1, 6], dtype=tf.float32)
    mask = tf.constant([[[1], [0]],
                        [[1], [1]]
                        ], dtype=tf.float32)
    # mask = tf.ones(shape=[2, 2, 1], dtype=tf.float32)

    _ = din_local_activation_block(user_behaviors=user_beh, mask=mask, query=query)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(fetches=tf.get_collection("print")))
