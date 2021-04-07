"""
https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/layers_cn/nn_cn.html#continuous-value-model
tensorflow implementation of continuous_value_model
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np


class ContinuousValueModel(object):
    """
    ContinuousValueModel: record the show and click of feature values
    forward: concat [log(show), log(ctr)] to the origin embedding of feature values
    backward: update the show and click of feature values according to the show_clicks parameter
    NOTE:
    Usage:
        cvm_layer = ContinuousValueModel(name="give_it_a_name", input_dim=100, output_dim=16)
        cvm_emb = cvm_layer(inp)
        .....
        show_clk_update_op = cvm_layer.update_show_clk(show_clicks)

        train_op = optimizer.Adam??
        train_op = tf.group(train_op, show_clk_update_op)

        ...
        sess.run(train_op)
    """

    def __init__(self, name, input_dim, output_dim):
        """
        :param name: string
        :param input_dim: int, number of distinct feature values
        :param output_dim: int, embedding size
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        with tf.variable_scope(name_or_scope=None, default_name="ContinuousValueModel"):
            self._emb_layer = tf.get_variable(name="{}_embedding".format(name),
                                              shape=[self._input_dim, self._output_dim],
                                              dtype=tf.float32,
                                              initializer=tf.initializers.glorot_normal)
            self._show_clk_layer = tf.get_variable(name="{}_show_clk".format(name),
                                                   shape=[self._input_dim, 2],
                                                   dtype=tf.float32,
                                                   trainable=False,
                                                   initializer=tf.initializers.ones)
            self._inp = None
        pass

    def __call__(self, inp, use_cvm=True):
        """
        :param inp: only support inp.shape = [None, 1] now
        :param use_cvm: boolean
        :return: use_cvm=True, [None, output_dim+2]; use_cvm=True, [None, output_dim]
        """
        assert len(inp.shape) == 2
        # assert inp.shape[1] == 1
        self._inp = inp
        emb = tf.nn.embedding_lookup(self._emb_layer, inp)
        show_clk_emb = tf.nn.embedding_lookup(self._show_clk_layer, inp)
        log_show = tf.log(show_clk_emb[:, :, 0:1])
        log_ctr = tf.log(show_clk_emb[:, :, 1:2]) - log_show
        show_clk_info = tf.concat([log_show, log_ctr], axis=2)
        if use_cvm:
            cvm = tf.concat([emb, show_clk_info], axis=2)
        else:
            cvm = emb
        # cvm = tf.squeeze(cvm, axis=1)
        return cvm

    def update_show_clk(self, show_clicks):
        """
        :param show_clicks: only support show_clicks.shape = [None, 2] now
        :return: update op, used to update the show click info of feature values
        """
        assert self._inp is not None, "call __call__ first!!"
        assert len(show_clicks.shape) == 2
        assert show_clicks.shape[1] == 2
        num_sub_fields = self._inp.shape[1]
        show_clicks = tf.tile(show_clicks, [1, num_sub_fields])
        inp = tf.reshape(self._inp, shape=[-1, 1])
        show_clicks = tf.reshape(show_clicks, shape=[-1, 2])
        # gradient map: shape [batch_size, num_input, 2]
        batch_size = tf.cast(tf.shape(inp)[0], dtype=tf.int64)
        batch_idx = tf.expand_dims(tf.range(start=0, limit=batch_size, dtype=tf.int64), axis=1)
        # [[[0, 9, 0], [0, 9, 1]], [[], []], ...]
        show_indices = tf.concat([batch_idx, inp, tf.zeros_like(inp)], axis=1)
        clk_indices = tf.concat([batch_idx, inp, tf.ones_like(inp)], axis=1)
        show_clk_indices = tf.stack([show_indices, clk_indices], axis=1)

        show_clk_grad_map = tf.scatter_nd(indices=show_clk_indices, updates=show_clicks,
                                          shape=[batch_size, self._input_dim, 2])
        show_clk_grad_map = tf.reduce_sum(show_clk_grad_map, axis=0)

        update_op = tf.assign_add(self._show_clk_layer, show_clk_grad_map)
        return update_op


if __name__ == '__main__':
    show_clicks = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    feats = tf.placeholder(dtype=tf.int64, shape=[None, 2])

    cvm_layer = ContinuousValueModel(name="test", input_dim=3, output_dim=3)
    inp_emb = cvm_layer(inp=feats, use_cvm=True)

    print(tf.global_variables())

    update = cvm_layer.update_show_clk(show_clicks)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        res = sess.run([inp_emb, update], feed_dict={feats: np.array([[0, 1], [2, 2], [0, 0]], dtype=np.int64),
                                                     show_clicks: np.array([[1, 1], [1, 0], [1, 0]],
                                                                           dtype=np.float)})

        print(res[0])

        res = sess.run([inp_emb, update], feed_dict={feats: np.array([[0, 1], [2, 2], [0, 0]], dtype=np.int64),
                                                     show_clicks: np.array([[1, 1], [1, 0], [1, 0]],
                                                                           dtype=np.float)})

        print(res[0])
