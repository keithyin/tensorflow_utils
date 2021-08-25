# coding=utf8
import tensorflow as tf


class MaskedMean(object):
    @staticmethod
    def masked_mean_3d(inp, mask):
        """
        :param inp: [b, T, num_fea, dim]
        :param mask: [b, T]
        :return: [b, num_fea, dim]
        """
        assert len(inp.shape) == 4, "invalid shape {}".format(inp)
        assert len(mask.shape) == 2, "invalid shape {}".format(mask)
        mask = tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1)
        # [b, T, num_fea, dim] - > [b, num_fea, dim]
        res = tf.reduce_sum(inp * mask, axis=1, keepdims=False)
        # [b, T, num_fea, 1] -> [b, num_fea, 1]
        length = tf.reduce_sum(mask, axis=1, keepdims=False)
        # in case that length = 0
        length = tf.where(length > 0, length, tf.ones_like(length))
        res = tf.div(res, length)
        return res

    @staticmethod
    def masked_mean_2d(inp, mask):
        """
        :param inp: [b, T, dim]
        :param mask: [b, T]
        :return: [b, dim]
        """
        # TODO: the code is same with masked_mean_3d
        assert len(inp.shape) == 3, "invalid shape {}".format(inp)
        assert len(mask.shape) == 2, "invalid shape {}".format(mask)
        mask = tf.expand_dims(mask, axis=-1)
        # [b, dim]
        res = tf.reduce_sum(inp * mask, axis=1, keepdims=False)
        # [b, 1]
        length = tf.reduce_sum(mask, axis=1, keepdims=False)
        # in case that length = 0
        length = tf.where(length > 0, length, tf.ones_like(length))
        res = tf.div(res, length)
        return res