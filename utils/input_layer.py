# coding=utf8
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import sparse_tensor
from ..utils.masked_mean import MaskedMean


class FeaProcessor(object):
    @staticmethod
    def var_len_fea_lookup(inp, pad_val, fea_num=1, lookup_table=None, emb_layer=None):
        """
        :param inp: SparseTensor or DenseTensor.
        :param fea_num:
        :param lookup_table:
        :param emb_layer:
        :param pad_val: pad value of var len
        :return: tuple: (inp, mask)
            inp: fea_num = 1 -> [b, T, emb_size], fea_num > 1 -> [b, T, fea_num, emb_size]
            mask: fea_num = 1 -> [b, T], fea_num > 1 -> [b, T]
        """
        out, mask = FeaProcessor.var_len_fea_lookup_v2(inp, pad_val, fea_num, lookup_table, emb_layer)
        if len(mask.shape) == 3:
            mask = tf.reduce_mean(mask, axis=-1, keep_dims=False, name="mask")
        return out, mask

    @staticmethod
    def var_len_fea_lookup_v2(inp, pad_val, fea_num=1, lookup_table=None, emb_layer=None):
        """
        The only difference from var_len_fea_lookup is mask.shape
        :param inp: SparseTensor or DenseTensor.
        :param fea_num:
        :param lookup_table:
        :param emb_layer:
        :param pad_val: pad value of var len
        :return: tuple: (inp, mask)
            out: fea_num = 1 -> [b, T, emb_size], fea_num > 1 -> [b, T, fea_num, emb_size]
            mask: mask.shape = out.shape[:-1]
        """
        if isinstance(inp, sparse_tensor.SparseTensor):
            inp = tf.sparse.to_dense(inp, default_value=pad_val)
        if fea_num > 1:
            assert inp.shape[-1] % fea_num == 0
            inp = tf.reshape(inp, shape=[tf.shape(inp)[0], inp.shape[-1] // fea_num, fea_num])
        mask = tf.cast(tf.not_equal(inp, pad_val), dtype=tf.float32, name="mask")

        if lookup_table is not None:
            inp = lookup_table.lookup(inp)
        if emb_layer is not None:
            assert not isinstance(emb_layer, tf.keras.layers.Embedding), "don't support keras"
            # first_emb_ = emb_layer[0] if isinstance(emb_layer, list) else emb_layer

            # if isinstance(first_emb_, tf.python.ops.lookup_ops.MutableDenseHashTableV2):
            #     inp = tf.nn.embedding_lookup_hashtable_v2(emb_layer, inp)
            # else:

            # NOTE: 有一些特征的默认值为-1，但是对于该 op 来说，如果特征值为-1的话，会报错，所以就在这 hard code 一下了。
            inp += 1
            try:
                #
                inp = tf.nn.embedding_lookup(emb_layer, inp)
            except:
                inp = tf.nn.embedding_lookup_hashtable_v2(emb_layer, inp, threshold=50)
        out = inp
        return out, mask

    @staticmethod
    def fix_len_fea_lookup(inp, lookup_table=None, emb_layer=None):
        """
        :return [*inp.shape, emb_size]
        """
        assert len(inp.shape) == 2, "fix_len_fea_lookup"
        if lookup_table is not None:
            inp = lookup_table.lookup(inp)
        if emb_layer is not None:
            assert not isinstance(emb_layer, tf.keras.layers.Embedding), "don't support keras"
            # NOTE: 有一些特征的默认值为-1，但是对于该 op 来说，如果特征值为-1的话，会报错，所以就在这 hard code 一下了。
            inp += 1
            try:
                inp = tf.nn.embedding_lookup(emb_layer, inp)
            except:
                inp = tf.nn.embedding_lookup_hashtable_v2(emb_layer, inp, threshold=50)
        return inp

    @staticmethod
    def var_len_fea_process(inp, fea_num=1, lookup_table=None, emb_layer=None, pad_val=0):
        """
        lookup and mean pooling
        :param inp: SparseTensor, [b, var_len]
        :param fea_num:
            for example: instance has a field named weather, if weather only contains temperature,humidity
            the fea_num is 2
        :param lookup_table: lookup_table, string -> id
        :param emb_layer: embedding layer, id -> emb
        :param pad_val: pad_val
        :return fea_num>1: [b, fea_num, emb_size]; fea_num=1:[b, emb_size]
        """
        emb, mask = FeaProcessor.var_len_fea_lookup(inp, pad_val, fea_num, lookup_table, emb_layer)
        if fea_num > 1:
            emb = MaskedMean.masked_mean_3d(emb, mask)
        else:
            emb = MaskedMean.masked_mean_2d(emb, mask)
        return emb

    @staticmethod
    def fix_len_fea_process(inp, lookup_table=None, emb_layer=None, aggr_method="flatten"):
        """
        lookup and mean pooling
        :param inp: Tensor, [b, fix_len]
        :param lookup_table: lookup_table, string -> id
        :param emb_layer: embedding layer, id -> emb
        :param aggr_method: "flatten"/"mean"/None
        :return
        """
        emb = FeaProcessor.fix_len_fea_lookup(inp, lookup_table, emb_layer)
        if aggr_method is None:
            pass
        elif aggr_method == "flatten":
            emb = tf.reshape(emb, shape=[-1, np.prod(emb.shape[1:])])
        elif aggr_method == "mean":
            emb = tf.reduce_mean(emb, axis=1)
        else:
            raise ValueError("aggr_method must be in 'flatten'/'mean'/None")
        return emb