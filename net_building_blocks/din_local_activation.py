from __future__ import print_function

import tensorflow as tf
import numpy as np


def din_local_activation_block(seq_feat, mask, query, activation_unit_dim=128, affine_dim=None):
    """
    :param seq_feat: [b, T, sub_field, dim] or [b, T, dim]
    :param mask: [b, T, sub_field] or [b, T]
    :param query: [b, num_sub_field, emb_size] or [b, emb_size]
    :param activation_unit_dim, int value.
    :param affine_dim: int. user_behavior and query to affine_dim
    :return: [b, some_dim]
    """
    with tf.variable_scope(name_or_scope=None, default_name="DinLocalActivationBlock"):
        batch_size = tf.shape(seq_feat)[0]
        time_step = tf.shape(seq_feat)[1]

        seq_feat = tf.reshape(seq_feat, shape=[batch_size, time_step, np.prod(seq_feat.shape[2:])])
        query = tf.reshape(query, shape=[batch_size, np.prod(query.shape[1:])])
        if affine_dim is not None:
            seq_feat = tf.layers.dense(seq_feat, units=affine_dim, activation=tf.nn.relu)
            query = tf.layers.dense(query, units=affine_dim, activation=tf.nn.relu)

        if len(mask.shape) == 3:
            mask = tf.reduce_mean(mask, axis=2, keepdims=True)
        elif len(mask.shape) == 2:
            mask = tf.expand_dims(mask, axis=2)
        else:
            raise ValueError("invalid mask dim. expected 2 or 3, but got {}".format(len(mask.shape)))

        # user_behaviors: [b, T, hidden_size], mask: [b, T, 1], query: [b, hidden_size]
        assert len(seq_feat.shape) == 3
        assert len(mask.shape) == 3
        assert len(query.shape) == 2

        assert seq_feat.shape[2] == query.shape[1], """
            user_behavior and query must share the same hidden_size, 
            but got user_behavior:{}, query:{}""".format(seq_feat, query)

        # [b, 1, hidden_size]
        query = tf.expand_dims(query, axis=1)
        query = tf.tile(query, multiples=[1, time_step, 1])

        # [b, T, hidden_size + hidden_size + hidden_size]
        state = tf.concat([query, seq_feat, query * seq_feat], axis=2)

        x = tf.layers.dense(state, activation_unit_dim, activation=tf.nn.relu)

        # [b, T, 1]
        weights = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, use_bias=False)
        weights = tf.multiply(weights, mask, name="weights")

        activated_behavior = tf.reduce_sum(weights * seq_feat, axis=1, keepdims=False)
    return activated_behavior


if __name__ == '__main__':
    user_beh = tf.random_uniform(shape=[2, 2, 1, 6], dtype=tf.float32)

    query = tf.random_uniform(shape=[2, 1, 6], dtype=tf.float32)
    mask = tf.constant([[[1], [0]],
                        [[1], [1]]
                        ], dtype=tf.float32)
    # mask = tf.ones(shape=[2, 2, 1], dtype=tf.float32)

    _ = din_local_activation_block(seq_feat=user_beh, mask=mask, query=query)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(fetches=tf.get_collection("print")))
